import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import cv2
import scipy.io as scio
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils import *
from .func_pfm import read_pfm


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                            str(args.scale_factor) + 'x/'
        # elif args.task == 'RE':
        #     self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
        #                        str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            Lr_SAI_y, Hr_SAI_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    data_list = None
    if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
        # if args.task == 'sr_model':
        dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                        str(args.scale_factor) + 'x/'
        data_list = os.listdir(dataset_dir)
        # elif args.task == 'RE':
        #     dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
        #                   str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
        #     data_list = os.listdir(dataset_dir)
    else:
        data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name))
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        # if args.task == 'sr_model':
        self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                            str(args.scale_factor) + 'x/'
        self.data_list = [data_name]
        # elif args.task == 'RE':
        #     self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
        #                        str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
        #     self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


class EntireLFDataSet(Dataset):
    def __init__(self, root_path, mode, angres, factor, dataset=['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']):
        super(EntireLFDataSet, self).__init__()
        self.data_list = []
        for data in dataset:
            self.data_list.extend([os.path.join(root_path, data, mode, name) for name in os.listdir(os.path.join(root_path, data, mode))])
        self.disparity = os.path.join(root_path, 'disparity')
        self.angres = angres
        self.factor = factor
        self.with_disparity = False

    def load_light_field(self, file_path):
        try:
            data = h5py.File(file_path, 'r')
            LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
        except:
            data = scio.loadmat(file_path)
            LF = np.array(data['LF'])

        (U, V, H, W, _) = LF.shape
        while H%self.factor != 0:
            H-=1
        while W%self.factor != 0:
            W-=1
        # Extract central angRes * angRes views
        LF = LF[(U-self.angres)//2:(U+self.angres)//2, (V-self.angres)//2:(V+self.angres)//2, :H, :W, 0:3]
        LF = LF.astype('double')
        return LF

    def downsample(self, LF):
        U, V, H, W, C = LF.shape
        res = np.zeros((U, V, H//self.factor, W//self.factor, C))
        for u in range(U):
            for v in range(V):
                # res[u, v] = F.interpolate(LF[u, v, :, :, :], 1/self.factor)
                res[u, v] = cv2.resize(LF[u, v, :, :, :], (W//self.factor, H//self.factor), interpolation=cv2.INTER_CUBIC)
        return res

    def load_disparity(self, path):
        res = read_pfm(path)
        H, W = res.shape
        while H%self.factor != 0:
            H-=1
        while W%self.factor != 0:
            W-=1
        return res[:H, :W]

    def rgb2ycrcb(self, LF):
        res = np.zeros_like(LF)
        res[:,:,0] =  65.481 * LF[:, :, 0] + 128.553 * LF[:, :, 1] +  24.966 * LF[:, :, 2] +  16.0
        res[:,:,1] = -37.797 * LF[:, :, 0] -  74.203 * LF[:, :, 1] + 112.000 * LF[:, :, 2] + 128.0
        res[:,:,2] = 112.000 * LF[:, :, 0] -  93.786 * LF[:, :, 1] -  18.214 * LF[:, :, 2] + 128.0
        return res/255.0

    def __getitem__(self, index):
        path = self.data_list[index]
        disparity_path = os.path.join(self.disparity, os.path.basename(path).replace('.mat', '.pfm'))
        if os.path.exists(disparity_path):
            disparity = self.load_disparity(disparity_path)
        else:
            disparity = None
        light_field = self.load_light_field(path)
        LR = self.downsample(light_field)
        data = rgb2ycbcr(LR)[0]
        label = rgb2ycbcr(light_field)[0]
        return data, label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    root = "/vepfs/Perception/Users/jianfei/self_exp/datasets/"
    dataset = EntireLFDataSet(root, "training", 5, 4)
    for idx, (data, label) in enumerate(dataset):
        print(data.shape)
        print(label.shape)
