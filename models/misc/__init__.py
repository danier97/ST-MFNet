from .gridnet import MIMOGridNet
from .pwcnet import Network as PWCNet
import torch
import torch.nn as nn
import torch.nn.functional as F




class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()
    def forward(self, x):
        return x




class Upsampler_8tap(nn.Module):
    def __init__(self):
        super(Upsampler_8tap, self).__init__()
        filt_8tap = torch.tensor([[-1,4,-11,40,40,-11,4,-1]]).div(64)
        self.filter = nn.Parameter(filt_8tap.repeat(3,1,1,1), requires_grad=False)

    def forward(self, im):
        b, c, h, w = im.shape
        im_up = torch.zeros(b,c,h*2,w*2).to(im.device)
        im_up[:,:,::2,::2] = im

        p = (8-1)//2
        im_up_row = F.conv2d(F.pad(im, pad=(p,p+1,0,0), mode='reflect'), self.filter, groups=3)
        im_up[:,:,0::2,1::2] = im_up_row
        im_up_col = torch.transpose(F.conv2d(F.pad(torch.transpose(im,2,3), pad=(p,p+1,0,0), mode='reflect'), self.filter, groups=3), 2, 3)
        im_up[:,:,1::2,0::2] = im_up_col
        im_up_cross = F.conv2d(F.pad(im_up[:,:,1::2,::2], pad=(p,p+1,0,0), mode='reflect'), self.filter, groups=3)
        im_up[:,:,1::2,1::2] = im_up_cross
        return im_up
