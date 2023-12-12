""" Model architecture of MT-HCCAR. """
from .cross_att import *

class MTHCCAR(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self, input_size):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(input_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
    )
      
    self.cls1 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid(),
    )
    self.cls2 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid(),
    )
    self.cls31 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
    )
    self.cls32 = nn.Sequential (
      nn.Linear(128, 3),
      nn.Sigmoid(),
    )

    self.reg1 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
    )
    self.cross_attn = nn.Sequential(
      CrossATT(in_channels=128),
    )
    self.reg2 = nn.Sequential (
      nn.Linear(128, 1),
    )

    self.recon = nn.Sequential (
      nn.Linear(32, 128),
      nn.ReLU(),
      nn.Linear(128, input_size),
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    feature = self.encoder(x)
    classification1 = self.cls1(feature)
    mask_cloudy = torch.where(classification1<=0.5, 1, 0)
    x_cloudy =  feature * mask_cloudy
    classification2 = self.cls2(x_cloudy)
    class3_feature = self.cls31(x_cloudy)
    classification3 = self.cls32(class3_feature)
    regression_feature = self.reg1(x_cloudy)
    attn_feature = self.cross_attn((regression_feature, class3_feature))
    regression = self.reg2(attn_feature)
    reconstruction = self.recon(feature)

    return reconstruction, classification1, classification2, classification3, mask_cloudy, regression