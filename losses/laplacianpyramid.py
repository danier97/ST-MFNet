import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianConv(nn.Module):
    def __init__(self):
        super(GaussianConv, self).__init__()
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        self.kernel = nn.Parameter(kernel.div(256).repeat(3,1,1,1), requires_grad=False)

    def forward(self, x, factor=1):
        c, h, w = x.shape[1:]
        p = (self.kernel.shape[-1]-1)//2
        blurred = F.conv2d(F.pad(x, pad=(p,p,p,p), mode='reflect'), factor*self.kernel, groups=c)
        return blurred

class LaplacianPyramid(nn.Module):
    """
    Implementing "The Laplacian pyramid as a compact image code." Burt, Peter J., and Edward H. Adelson. 
    """
    def __init__(self, max_level=5):
        super(LaplacianPyramid, self).__init__()
        self.gaussian_conv = GaussianConv()
        self.max_level = max_level

    def forward(self, X):
        pyramid = []
        current = X
        for _ in range(self.max_level-1):
            blurred = self.gaussian_conv(current)
            reduced = self.reduce(blurred)
            expanded = self.expand(reduced)
            diff = current - expanded
            pyramid.append(diff)
            current = reduced

        pyramid.append(current)

        return pyramid
    
    def reduce(self, x):
        return F.avg_pool2d(x, 2)
    
    def expand(self, x):
        # injecting even zero rows
        tmp = torch.cat([x, torch.zeros_like(x).to(x.device)], dim=3)
        tmp = tmp.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
        tmp = tmp.permute(0,1,3,2)
        # injecting even zero columns
        tmp = torch.cat([tmp, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
        tmp = tmp.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
        x_up = tmp.permute(0,1,3,2)
        # convolve with 4 x Gaussian kernel
        return self.gaussian_conv(x_up, factor=4)

class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

        self.criterion = nn.L1Loss(reduction='sum')
        self.lap = LaplacianPyramid()

    def forward(self, x, y):
        x_lap, y_lap = self.lap(x), self.lap(y)
        return sum(2**i * self.criterion(a, b) for i, (a, b) in enumerate(zip(x_lap, y_lap)))
