import sys
import cv2 
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from utils import vis_mat
sys.path.append('..')
from model.LF_InterNet import InterNet

def vis_rawdata(path, save_path):
    vis_mat(path, save_path)

def vis_result(result, vis_path):
    cv2.imwrite(vis_path, result)

def read_H5(path):
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
    return data, label

def inference(net, data_path):
    net.eval()
    net.to(device='cuda:0')
    with h5py.File(data_path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        data = np.expand_dims(data, axis=0)
        data = np.expand_dims(data, axis=0)
        data = torch.from_numpy(data.copy())
        data = Variable(data).to('cuda:0')
    import time
    start = time.time()
    with torch.no_grad():
        out = net(data)
    print(f'inference spend time: {time.time()-start}')
    plt.subplot(121)
    plt.imshow(out.data.cpu().numpy())
    plt.subplot(122)
    plt.imshow(label)
    plt.show()
    return np.squeeze(out.data.cpu().numpy()), label

if __name__ == '__main__':
    net_raw = InterNet(angRes=5, n_blocks=4, n_layers=4, channels=64, upscale_factor=4)
    state_dict = torch.load('../log/InterNet_5x5_4xSR_C64.pth.tar')['state_dict']
    net_raw.load_state_dict(state_dict)
    out, label = inference(net_raw, data_path=r'D:\BaiduNetdiskDownload\Datasets\training\000002.h5')
    import pdb;pdb.set_trace()
