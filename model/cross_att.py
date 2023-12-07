import torch
from torch import nn
from torch.nn import functional as F


class CrossAtt(nn.Module):
    def __init__(self, in_channels):
        """
        args:
            in_channels: original channel size (1024 in the paper)
            dimension: can be 1, 2
        """
        super(CrossAtt, self).__init__()

        self.in_channels = in_channels
        # the channel size is reduced to half inside the block
        self.inter_channels = in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        # function g in the paper which goes through conv. with kernel size 1
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W_z = nn.Linear(self.inter_channels, self.in_channels)

        # By initializing Wz to 0, this block can be inserted to any existing architecture
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
            
    def forward(self, x):
        """
        args
            x1, x2 shape: (N, C, T) 
        """

        x1,x2 = x
        batch_size = x1.size(0)
        
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