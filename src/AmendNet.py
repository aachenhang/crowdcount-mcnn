import torch
import torch.nn as nn
import network
from network import Conv2d


class MCNN_BackBone(nn.Module):
    '''
    Multi-column CNN Head
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN_BackBone, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        #self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        
        return x


class MCNNNet(nn.Module):
    def __init__(self, mcnn_backbone=None, bn=False):
        super(MCNNNet, self).__init__()
        if(mcnn_backbone is not None):
            self.mcnn_backbone = mcnn_backbone
        else:
            self.mcnn_backbone = MCNN_BackBone()      
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))

        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, im_data, gt_data=None):        
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)                
        feature_map = self.mcnn_backbone(im_data)
        density_map = self.fuse(feature_map)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = nn.MSELoss()(density_map, gt_data)
            
        return density_map

    
class AmendNet(nn.Module):
    '''
    AmendNet
        -Based on MCNN and amend the density map
    '''
    def __init__(self, mcnn_backbone=None, bn=False):
        super(AmendNet, self).__init__()
        
        if(mcnn_backbone is not None):
            self.mcnn_backbone = mcnn_backbone
        else:
            self.mcnn_backbone = MCNN_BackBone()
        self.conv3x3 = Conv2d( 1, 30, 3, same_padding=True, bn=bn)
        self.downsample = nn.AvgPool2d(4)
        self.amend = nn.Sequential(Conv2d( 60, 50, 11, same_padding=True, bn=bn),
                                   Conv2d( 50, 40, 9, same_padding=True, bn=bn),
                                   Conv2d( 40, 30, 7, same_padding=True, bn=bn),
                                   Conv2d( 30, 30, 5, same_padding=True, bn=bn))
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))
            
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training) 
        feature_map = self.mcnn_backbone(im_data)
        x = self.conv3x3(im_data)
        x = self.downsample(x)
        x = torch.cat((feature_map, x), 1)
        x = self.amend(x)
        density_map = self.fuse(x)
        
        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = nn.MSELoss()(density_map, gt_data)
    
        return density_map
    
    
    
    