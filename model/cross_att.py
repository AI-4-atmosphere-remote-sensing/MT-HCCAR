import torch
from torch import nn
from torch.nn import functional as F


class CrossATT(nn.Module):
    def __init__(self, in_channels, inter_channels=None):

        super(CrossATT, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        else:
            self.inter_channels = inter_channels

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W_z = nn.Linear(self.inter_channels, self.in_channels)

        # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
            
    def forward(self, x):

        x1,x2 = x
        batch_size = x1.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x1).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x2).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x1).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x1.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x1

        return z
