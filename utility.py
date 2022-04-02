import torch
import numpy as np
import cv2
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

def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth):
    if bit_depth == 8:
        datatype = np.uint8
        stream.seek(iFrame*1.5*width*height)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    else:
        datatype = np.uint16
        stream.seek(iFrame*3*width*height)
        Y = np.fromfile(stream, dtype=datatype, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=datatype, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    
    yuv = np.empty((height*3//2, width), dtype=datatype)
    yuv[0:height,:] = Y

    yuv[height:height+height//4,:] = U.reshape(-1, width)
    yuv[height+height//4:,:] = V.reshape(-1, width)

    if bit_depth == 10:
        yuv = (yuv/1023*255).astype(np.uint8)

    #convert to rgb
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

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
