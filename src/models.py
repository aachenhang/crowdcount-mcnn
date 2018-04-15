import torch
import torch.nn as nn
from network import Conv2d, MSB_Conv

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
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
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x

    
class MSCNN(nn.Module):
    '''
    Multi-Scale CNN 
        -Implementation of Multi-Scale Convolutional Neural networks for crowd counting (Lingke Zeng et al.)
    '''
    
    def __init__(self, bn=False):
        super(MSCNN, self).__init__()
        self.network = nn.Sequential(Conv2d( 1, 64, 9, same_padding=True, bn=bn),
                                    MSB_Conv(64, 4*16, [9,7,5,3]),
                                    nn.MaxPool2d(2),
                                    MSB_Conv(64, 4*16, [9,7,5,3]),
                                    MSB_Conv(64, 4*16, [9,7,5,3]),
                                    nn.MaxPool2d(2),
                                    MSB_Conv(4*16, 3*32, [7,5,3]),
                                    MSB_Conv(3*32, 3*32, [7,5,3]),
                                    Conv2d( 3*32, 1000, 1, same_padding=True, bn=bn),
                                    Conv2d( 1000, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        x = self.network(im_data)
        return x
    

    
