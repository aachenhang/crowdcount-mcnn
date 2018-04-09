from crowd_count import CrowdCounter
from AmendNet import MCNNNet, AmendNet
import network
import numpy as np


def evaluate_model(trained_model, data_loader, netname='CrowdCounter'):
    if(netname is 'CrowdCounter'):
        net = CrowdCounter()
    elif(netname is 'MCNNNet'):
        net = MCNNNet()
    elif(netname is 'AmendNet'):
        net = AmendNet()
    else:
        raise('netname should be one of ["CrowdCounter", "MCNNNet", "AmendNet"],\
        but we got ', netname)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:                        
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))        
    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse

    