import torch.nn as nn
import network
from models import MCNN, MSCNN


class CrowdCounter(nn.Module):
    def __init__(self, mcnn=None):
        super(CrowdCounter, self).__init__()
        if(mcnn is not None):
            self.DME = mcnn
        else:
            self.DME = MCNN()        
#         self.loss_fn = nn.MSELoss()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, im_data, gt_data=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                
        density_map = self.DME(im_data)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = nn.MSELoss()(density_map, gt_data)
            
        return density_map
    
#     def build_loss(self, density_map, gt_data):
#         loss = self.loss_fn(density_map, gt_data)
#         return loss
        
        
class CrowdCounter_MSCNN(nn.Module):
    def __init__(self):
        super(CrowdCounter_MSCNN, self).__init__()
        self.msnn = MSCNN()
    
    @property
    def loss(self):
        return self.loss_mse
    
    
    def forward(self, im_data, gt_data=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                
        density_map = self.msnn(im_data)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = nn.MSELoss()(density_map, gt_data)
            
        return density_map
    
