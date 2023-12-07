""" Model architecture of MT-HCCAR. """
from cross_att import *

class MTHCCAR(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.encoder = (
      nn.Linear(input_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
    )
      
    self.cls_mask = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid(),
    )
    self.cls_phase = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid(),
    )
    self.cls_aux1 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
    )
    self.cls_aux2 = nn.Sequential (
      nn.Linear(128, 3),
      nn.Sigmoid(),
    )

    self.reg_cot1 = nn.Sequential (
      nn.Linear(32, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
    )
    self.cross_attn = nn.Sequential(
      # Cross att with a residual connection
      CrossAtt(in_channels=128, dimension=2),
    )
    self.reg_cot2 = nn.Sequential (
      nn.Linear(128, 1),
    )

    self.recon = nn.Sequential (
      nn.Linear(32, 128),
      nn.ReLU(),
      nn.Linear(128, input_size),
    )

  def forward(self, x):
    # Encoder
    feature = self.encoder(x)
    # HC
    cloud_mask = self.cls_mask(feature)
    mask_binary = torch.where(cloud_mask<=0.5, 1, 0)
    x_cloudy =  feature * mask_binary
    cloud_phase = self.cls_phase(x_cloudy)
    # CAR
    aux_feature = self.cls_aux1(x_cloudy)
    cls_aux = self.cls_aux2(aux_feature)
    regression_feature = self.reg_cot1(x_cloudy)
    attn_feature = self.cross_attn((regression_feature, cls_aux))
    regression = self.reg_cot2(attn_feature)
    # Decoder
    reconstruction = self.recon(feature)

    return reconstruction, cloud_mask, cloud_phase, cls_aux, mask_binary, regression