import torch
import torch.nn as nn
from models.misc.resnet_3D import r3d_18, Conv_3d, upConv3D
from models.misc import Identity
import cupy_module.adacof as adacof
from cupy_module.softsplat import ModuleSoftsplat
import sys
from torch.nn import functional as F
from utility import moduleNormalize, gaussian_kernel
from models import feature
from models.misc import MIMOGridNet, Upsampler_8tap
from models.misc import PWCNet
from models.misc.pwcnet import backwarp

class UNet3d_18(nn.Module):
    def __init__(self, channels=[32,64,96,128], bn=True):
        super(UNet3d_18, self).__init__()
        growth = 2 # since concatenating previous outputs
        upmode = "transpose" # use transposeConv to upsample

        self.channels = channels

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.encoder = r3d_18(bn=bn, channels=channels)

        self.decoder = nn.Sequential(
            Conv_3d(channels[::-1][0], channels[::-1][1] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][1]*growth, channels[::-1][2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            upConv3D(channels[::-1][2]*growth, channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode),
            Conv_3d(channels[::-1][3]*growth, channels[::-1][3] , kernel_size=3, padding=1, bias=True),
            upConv3D(channels[::-1][3]*growth , channels[::-1][3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode)
        )

        self.feature_fuse = nn.Sequential(
            *([nn.Conv2d(channels[::-1][3]*5, channels[::-1][3], kernel_size=1, stride=1, bias=False)] + \
              [nn.BatchNorm2d(channels[::-1][3]) if bn else Identity])
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels[::-1][3], 3 , kernel_size=7, stride=1, padding=0) 
        )
    
    def forward(self, im1, im3, im5, im7, im4_tilde):
        images = torch.stack((im1, im3, im4_tilde, im5, im7) , dim=2)

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3 , x_3], dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2 , x_2], dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1 , x_1], dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0 , x_0], dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        return out




class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )
        
        def Subnet_offset_ds(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
            )

        def Subnet_weight_ds(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        def Subnet_offset_us(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        def Subnet_weight_us(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.Softmax(dim=1)
            )

        self.moduleWeight1_ds = Subnet_weight_ds(self.kernel_size ** 2)
        self.moduleAlpha1_ds = Subnet_offset_ds(self.kernel_size ** 2)
        self.moduleBeta1_ds = Subnet_offset_ds(self.kernel_size ** 2)
        self.moduleWeight2_ds = Subnet_weight_ds(self.kernel_size ** 2)
        self.moduleAlpha2_ds = Subnet_offset_ds(self.kernel_size ** 2)
        self.moduleBeta2_ds = Subnet_offset_ds(self.kernel_size ** 2)

        self.moduleWeight1 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size ** 2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size ** 2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size ** 2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size ** 2)

        self.moduleWeight1_us = Subnet_weight_us(self.kernel_size ** 2)
        self.moduleAlpha1_us = Subnet_offset_us(self.kernel_size ** 2)
        self.moduleBeta1_us = Subnet_offset_us(self.kernel_size ** 2)
        self.moduleWeight2_us = Subnet_weight_us(self.kernel_size ** 2)
        self.moduleAlpha2_us = Subnet_offset_us(self.kernel_size ** 2)
        self.moduleBeta2_us = Subnet_offset_us(self.kernel_size ** 2)


    def forward(self, tensorCombine):
        # Frame 0
        Weight1_ds = self.moduleWeight1_ds(tensorCombine)
        Weight1 = self.moduleWeight1(tensorCombine)
        Weight1_us = self.moduleWeight1_us(tensorCombine)
        Alpha1_ds = self.moduleAlpha1_ds(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Alpha1_us = self.moduleAlpha1_us(tensorCombine)
        Beta1_ds = self.moduleBeta1_ds(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Beta1_us = self.moduleBeta1_us(tensorCombine)

        # Frame 2
        Weight2_ds = self.moduleWeight2_ds(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Weight2_us = self.moduleWeight2_us(tensorCombine)
        Alpha2_ds = self.moduleAlpha2_ds(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Alpha2_us = self.moduleAlpha2_us(tensorCombine)
        Beta2_ds = self.moduleBeta2_ds(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Beta2_us = self.moduleBeta2_us(tensorCombine)

        return Weight1_ds, Alpha1_ds, Beta1_ds, Weight2_ds, Alpha2_ds, Beta2_ds, \
               Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2, \
               Weight1_us, Alpha1_us, Beta1_us, Weight2_us, Alpha2_us, Beta2_us


class STMFNet(torch.nn.Module):
    def __init__(self, args):

        super(STMFNet, self).__init__()
        class Metric(torch.nn.Module):
            def __init__(self):
                super(Metric, self).__init__()
                self.paramScale = torch.nn.Parameter(-torch.ones(1, 1, 1, 1))
            def forward(self, tenFirst, tenSecond, tenFlow):
                return self.paramScale * F.l1_loss(input=tenFirst, target=backwarp(tenSecond, tenFlow), reduction='none').mean(1, True)

        self.args = args
        self.kernel_size = args.kernel_size
        self.kernel_pad = int(((args.kernel_size - 1) * args.dilation) / 2.0)
        self.dilation = args.dilation

        self.feature_extractor = getattr(feature, args.featnet)(args.featc, norm_layer=args.featnorm)

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

        self.moduleAdaCoF = adacof.FunctionAdaCoF.apply

        self.gauss_kernel = torch.nn.Parameter(gaussian_kernel(5, 0.5).repeat(3,1,1,1), requires_grad=False)

        self.upsampler = Upsampler_8tap()

        self.scale_synthesis = MIMOGridNet((6,6+6,6), (3,), grid_chs=(32, 64, 96), n_row=3, n_col=4, outrow=(1,))

        self.flow_estimator = PWCNet()

        self.softsplat = ModuleSoftsplat(strType='softmax')

        self.metric = Metric()

        self.dyntex_generator = UNet3d_18(bn=args.featnorm)

        # freeze weights of PWCNet if not finetuning it
        if not args.finetune_pwc:
            for param in self.flow_estimator.parameters():
                param.requires_grad = False

    def forward(self, I0, I1, I2, I3, *args):
        h0 = int(list(I1.size())[2])
        w0 = int(list(I1.size())[3])
        h2 = int(list(I2.size())[2])
        w2 = int(list(I2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 128 != 0:
            pad_h = 128 - (h0 % 128)
            I0 = F.pad(I0, (0, 0, 0, pad_h), mode='reflect')
            I1 = F.pad(I1, (0, 0, 0, pad_h), mode='reflect')
            I2 = F.pad(I2, (0, 0, 0, pad_h), mode='reflect')
            I3 = F.pad(I3, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 128 != 0:
            pad_w = 128 - (w0 % 128)
            I0 = F.pad(I0, (0, pad_w, 0, 0), mode='reflect')
            I1 = F.pad(I1, (0, pad_w, 0, 0), mode='reflect')
            I2 = F.pad(I2, (0, pad_w, 0, 0), mode='reflect')
            I3 = F.pad(I3, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        feats = self.feature_extractor(moduleNormalize(I1), moduleNormalize(I2))
        kernelest = self.get_kernel(feats)
        Weight1_ds, Alpha1_ds, Beta1_ds, Weight2_ds, Alpha2_ds, Beta2_ds = kernelest[:6]
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2 = kernelest[6:12]
        Weight1_us, Alpha1_us, Beta1_us, Weight2_us, Alpha2_us, Beta2_us = kernelest[12:]

        # Original scale
        tensorAdaCoF1 = self.moduleAdaCoF(self.modulePad(I1), Weight1, Alpha1, Beta1, self.dilation)*1.
        tensorAdaCoF2 = self.moduleAdaCoF(self.modulePad(I2), Weight2, Alpha2, Beta2, self.dilation)*1.

        # 1/2 downsampled version
        c, h, w = I1.shape[1:]
        p = (self.gauss_kernel.shape[-1]-1)//2
        I1_blur = F.conv2d(F.pad(I1, pad=(p,p,p,p), mode='reflect'), self.gauss_kernel, groups=c)
        I2_blur = F.conv2d(F.pad(I2, pad=(p,p,p,p), mode='reflect'), self.gauss_kernel, groups=c)
        I1_ds = F.interpolate(I1_blur, size=(h//2,w//2), mode='bilinear', align_corners=False)
        I2_ds = F.interpolate(I2_blur, size=(h//2,w//2), mode='bilinear', align_corners=False)
        tensorAdaCoF1_ds = self.moduleAdaCoF(self.modulePad(I1_ds), Weight1_ds, Alpha1_ds, Beta1_ds, self.dilation)*1.
        tensorAdaCoF2_ds = self.moduleAdaCoF(self.modulePad(I2_ds), Weight2_ds, Alpha2_ds, Beta2_ds, self.dilation)*1.

        # x2 upsampled version
        I1_us = self.upsampler(I1)
        I2_us = self.upsampler(I2)
        tensorAdaCoF1_us = self.moduleAdaCoF(self.modulePad(I1_us), Weight1_us, Alpha1_us, Beta1_us, self.dilation)*1.
        tensorAdaCoF2_us = self.moduleAdaCoF(self.modulePad(I2_us), Weight2_us, Alpha2_us, Beta2_us, self.dilation)*1.

        # use softsplat for refinement
        pyramid0, pyramid2 = self.flow_estimator.extract_pyramid(I1, I2)
        flow_0_2 = 20*self.flow_estimator(I1, I2, pyramid0, pyramid2)
        flow_0_2 = F.interpolate(flow_0_2, size=(h,w), mode='bilinear', align_corners=False)
        flow_2_0 = 20*self.flow_estimator(I2, I1, pyramid2, pyramid0)
        flow_2_0 = F.interpolate(flow_2_0, size=(h,w), mode='bilinear', align_corners=False)
        metric_0_2 = self.metric(I1, I2, flow_0_2)
        metric_2_0 = self.metric(I2, I1, flow_2_0)
        tensorSoftsplat0 = self.softsplat(I1, 0.5*flow_0_2, metric_0_2)
        tensorSoftsplat2 = self.softsplat(I2, 0.5*flow_2_0, metric_2_0)

        # synthesize multiple scales
        tensorCombine_us = torch.cat([tensorAdaCoF1_us, tensorAdaCoF2_us], dim=1)
        tensorCombine = torch.cat([tensorAdaCoF1, tensorAdaCoF2, tensorSoftsplat0, tensorSoftsplat2], dim=1)
        tensorCombine_ds = torch.cat([tensorAdaCoF1_ds, tensorAdaCoF2_ds], dim=1)
        output_tilde = self.scale_synthesis(tensorCombine_us, tensorCombine, tensorCombine_ds)[0]

        # generate dynamic texture
        dyntex = self.dyntex_generator(I0,I1,I2,I3,output_tilde)
        output = output_tilde + dyntex

        if h_padded:
            output = output[:, :, 0:h0, :]
        if w_padded:
            output = output[:, :, :, 0:w0]

        return output