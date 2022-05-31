import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from metrics import pytorch_ssim


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )
    elif args.decay_type == 'plateau':
        scheduler = lrs.ReduceLROnPlateau(
            my_optimizer,
            mode='max',
            factor=args.gamma,
            patience=args.patience,
            threshold=0.01, # metric to be used is psnr
            threshold_mode='abs',
            verbose=True
        )

    return scheduler


def gaussian_kernel(sz, sigma):
    k = torch.arange(-(sz-1)/2, (sz+1)/2)
    k = torch.exp(-1.0/(2*sigma**2) * k**2)
    k = k.reshape(-1, 1) * k.reshape(1, -1)
    k = k / torch.sum(k)
    return k


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)], 1)


class FoldUnfold():
    '''
    Class to handle folding tensor frame into batch of patches and back to frame again
    Thanks to Charlie Tan (charlie.tan.2019@bristol.ac.uk) for the earier version.
    '''

    def __init__(self, height, width, patch_size, overlap):

        if height % 2 or width % 2 or patch_size % 2 or overlap % 2:
            print("only defined for even values of height, width, patch_size size and overlap, odd values will reconstruct incorrectly")
            return

        self.height = height
        self.width = width

        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def fold_to_patches(self, *frames):
        '''
        args: frames -- list of (1,3,H,W) tensors
        returns: list of (B,3,h,w) image patches
        '''

        # number of blocks in each direction
        n_blocks_h = (self.height // (self.stride)) + 1 
        n_blocks_w = (self.width // (self.stride)) + 1 
    
        # how much to pad each edge by 
        self.pad_h = (self.stride * n_blocks_h + self.overlap - self.height) // 2
        self.pad_w = (self.stride * n_blocks_w + self.overlap - self.width) // 2
        self.height_pad = self.height + 2*self.pad_h
        self.width_pad = self.width + 2*self.pad_w
    
        # pad the frames and unfold into patches
        patches_list = []
        for i in range(len(frames)):
            padded = F.pad(frames[i], (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode='reflect')
            unfolded = F.unfold(padded, self.patch_size, stride=self.stride)
            patches = unfolded.permute(2, 1, 0).reshape(-1, 3, self.patch_size, self.patch_size)
            patches_list.append(patches)
 
        return patches_list
    
    def unfold_to_frame(self, patches):
        '''
        args: patches -- tensor of shape (B,3,h,w)
        returns: frame -- tensor of shape (1,3,H,W)
        '''

        # reshape and permute back into [frames, chans * patch_size ** 2, num_patches] as expected by fold
        frame_unfold = patches.reshape(-1, 3 * self.patch_size ** 2, 1).permute(2, 1, 0)
    
        # fold into tensor of shape pad_shape
        frame_fold = F.fold(frame_unfold, (self.height_pad, self.width_pad), self.patch_size, stride=self.stride)
    
        # unfold sums overlaps instead of averaging so tensor of ones unfolded and
        # folded to track overlaps and take mean of overlapping pixels
        ones = torch.ones_like(frame_fold)
        ones_unfold = F.unfold(ones, self.patch_size, stride=self.stride)
    
        # divisor is tensor of shape pad_shape where each element is the number of values that have overlapped
        # 1 = no overlaps
        divisor = F.fold(ones_unfold, (self.height_pad, self.width_pad), self.patch_size, stride=self.stride)
    
        # divide reconstructed frame by divisor
        frame_div = frame_fold / divisor
    
        # crop frame to remove the padded areas
        frame_crop = frame_div[:,:,self.pad_h:-self.pad_h,self.pad_w:-self.pad_w].clone()
    
        return frame_crop


def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth, pix_fmt='420'):
    if pix_fmt == '420':
        multiplier = 1
        uv_factor = 2
    elif pix_fmt == '444':
        multiplier = 2
        uv_factor = 1
    else:
        print('Pixel format {} is not supported'.format(pix_fmt))
        return

    if bit_depth == 8:
        datatype = np.uint8
        stream.seek(iFrame*1.5*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    else:
        datatype = np.uint16
        stream.seek(iFrame*3*width*height*multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))
        V = np.fromfile(stream, dtype=datatype, count=(width//uv_factor)*(height//uv_factor)).\
                                reshape((height//uv_factor, width//uv_factor))

    if pix_fmt == '420':
        yuv = np.empty((height*3//2, width), dtype=datatype)
        yuv[0:height,:] = Y

        yuv[height:height+height//4,:] = U.reshape(-1, width)
        yuv[height+height//4:,:] = V.reshape(-1, width)

        if bit_depth != 8:
            yuv = (yuv/(2**bit_depth-1)*255).astype(np.uint8)

        #convert to rgb
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    
    else:
        yvu = np.stack([Y,V,U],axis=2)
        if bit_depth != 8:
            yvu = (yvu/(2**bit_depth-1)*255).astype(np.uint8)
        rgb = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)

    return rgb


def quantize(imTensor):
    return imTensor.clamp(0.0, 1.0).mul(255).round()


def tensor2rgb(tensor):
    """
    Convert GPU Tensor to RGB image (numpy array)
    """
    out = []
    for b in range(tensor.shape[0]):
        out.append(np.moveaxis(quantize(tensor[b]).cpu().detach().numpy(), 0, 2).astype(np.uint8))
    return np.array(out) #(B,H,W,C)


def calc_psnr(gt, out):
    """
    args:
    gt, out -- (B,3,H,W) cuda Tensors
    """
    mse = torch.mean((quantize(gt) - quantize(out))**2, dim=(1,2,3))
    return -10 * torch.log10(mse/255**2 + 1e-8) # (B,)


def calc_ssim(gt, out, size_average=False):
    return pytorch_ssim.ssim_matlab(quantize(gt), quantize(out), size_average=size_average)
