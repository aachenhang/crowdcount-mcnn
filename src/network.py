import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    
class MSB_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_1, kernel_size_2, kernel_size_3, kernel_size_4 = None, stride=1, relu=True, same_padding=False, bn=False):
        super(MSB_Conv, self).__init__()
        if kernel_size_4 is not None:
            out_channels = int(out_channels / 4)
        else:
            out_channels = int(out_channels / 3)
        padding_1 = int((kernel_size_1 - 1) / 2) if same_padding else 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size_1, stride, padding=padding_1)
        padding_2 = int((kernel_size_2 - 1) / 2) if same_padding else 0
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size_2, stride, padding=padding_2)
        padding_3 = int((kernel_size_3 - 1) / 2) if same_padding else 0
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size_3, stride, padding=padding_3)
        if kernel_size_4 is not None:
            padding_4 = int((kernel_size_4 - 1) / 2) if same_padding else 0
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size_4, stride, padding=padding_4)
        else:
            self.conv4 = None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        if self.conv4 is None:
            x = torch.cat((x1, x2, x3),1)
        else:
            x4 = self.conv4(x)
            x = torch.cat((x1, x2, x3, x4),1)
        
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
    
def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net, prefix=''):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[prefix+k]))         
        v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
