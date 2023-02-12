import argparse
import os
import h5py
from pathlib import Path
import scipy.io as scio
import sys
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--angRes", type=int, default=7, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")
    parser.add_argument('--mode', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='/vepfs/Perception/Users/jianfei/self_exp/datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='/vepfs/Perception/Users/jianfei/self_exp/datasets', help='')

    return parser.parse_args()


def generate_training_data(args):
    angRes, scale_factor = args.angRes, args.scale_factor
    patchsize = scale_factor * 32
    stride = patchsize // 2
    downRatio = 1 / scale_factor

    ''' dir '''
    save_dir = os.path.join(args.save_data_path, 'data_for_training')
    save_dir = os.path.join(save_dir, 'SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    os.makedirs(save_dir, exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = os.path.join(save_dir, name_dataset)
        os.makedirs(sub_save_dir, exist_ok=True)

        src_sub_dataset = os.path.join(args.src_data_path, name_dataset, args.mode)
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(os.path.join(root, file), 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(os.path.join(root, file))
                    LF = np.array(data['LF'])

                (U, V, _, _, _) = LF.shape
                # Extract central angRes * angRes views
                LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, :, :, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                for h in range(0, H - patchsize + 1, stride):
                    for w in range(0, W - patchsize + 1, stride):
                        idx_save = idx_save + 1
                        idx_scene_save = idx_scene_save + 1
                        Hr_SAI_y = np.zeros((U * patchsize, V * patchsize),dtype='single')
                        Lr_SAI_y = np.zeros((U * patchsize // scale_factor, V * patchsize // scale_factor),dtype='single')

                        for u in range(U):
                            for v in range(V):
                                tmp_Hr_rgb = LF[u, v, h: h + patchsize, w: w + patchsize,:]
                                tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb) 
                                tmp_Hr_y = tmp_Hr_ycbcr[:, :, 0]

                                patchsize_Lr = patchsize // scale_factor
                                Hr_SAI_y[u * patchsize : (u+1) * patchsize, v * patchsize: (v+1) * patchsize] = tmp_Hr_y
                                tmp_Sr_y = imresize(tmp_Hr_y, scalar_scale=downRatio)

                                Lr_SAI_y[u*patchsize_Lr : (u+1)*patchsize_Lr, v*patchsize_Lr: (v+1)*patchsize_Lr] = tmp_Sr_y

                        # save
                        file_name = [str(sub_save_dir) + '/' + '%06d'%idx_save + '.h5']
                        with h5py.File(file_name[0], 'w') as hf:
                            hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                            hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')
                            hf.close()
                print('%d training samples have been generated\n' % (idx_scene_save))

def generate_test_data(args):
    angRes, scale_factor = args.angRes, args.scale_factor
    downRatio = 1 / scale_factor

    ''' dir '''
    save_dir = os.path.join(args.save_data_path, 'data_for_test')
    save_dir = os.path.join(save_dir, 'SR_' + str(angRes) + 'x' + str(angRes) + '_' + str(scale_factor) + 'x')
    os.makedirs(save_dir, exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = os.path.join(save_dir, name_dataset)
        os.makedirs(sub_save_dir, exist_ok=True)

        src_sub_dataset = os.path.join(args.src_data_path, name_dataset, args.mode)
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(os.path.join(root, file), 'r')
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(os.path.join(root, file))
                    LF = np.array(data['LF'])

                (U, V, H, W, _) = LF.shape
                H = H // 4 * 4
                W = W // 4 * 4

                # Extract central angRes * angRes views
                LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                Hr_SAI_y = np.zeros((U * H, V * W), dtype='single')
                Sr_SAI_cbcr = np.zeros((U * H, V * W, 2), dtype='single')
                Lr_SAI_y = np.zeros((U * H // scale_factor, V * W // scale_factor), dtype='single')

                for u in range(U):
                    for v in range(V):
                        tmp_Hr_rgb = LF[u, v, :, :, :]
                        tmp_Hr_ycbcr = rgb2ycbcr(tmp_Hr_rgb)
                        Hr_SAI_y[u * H: (u+1) * H, v * W: (v+1)* W] = tmp_Hr_ycbcr[:, :, 0]

                        tmp_Hr_y = tmp_Hr_ycbcr[:, :, 0]
                        tmp_Lr_y = imresize(tmp_Hr_y, scalar_scale=downRatio)
                        Lr_SAI_y[u*H//scale_factor : (u+1)*H//scale_factor, v*W//scale_factor : (v+1)*W//scale_factor] = tmp_Lr_y

                        tmp_Hr_cbcr = tmp_Hr_ycbcr[:,:,1:3]
                        tmp_Lr_cbcr = imresize(tmp_Hr_cbcr, scalar_scale=downRatio)
                        tmp_Sr_cbcr = imresize(tmp_Lr_cbcr, scalar_scale=scale_factor)
                        Sr_SAI_cbcr[u * H: (u+1) * H, v * W: (v+1)* W,:] = tmp_Sr_cbcr

                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    hf.create_dataset('Lr_SAI_y', data=Lr_SAI_y.transpose((1, 0)), dtype='single')
                    hf.create_dataset('Sr_SAI_cbcr', data=Sr_SAI_cbcr.transpose((2, 1, 0)), dtype='single')
                    hf.create_dataset('Hr_SAI_y', data=Hr_SAI_y.transpose((1, 0)), dtype='single')
                    hf.close()

                print('%d test samples have been generated\n' % (idx_scene_save))


if __name__ == '__main__':
    args = parse_args()
    if args.mode == "training":
        generate_training_data(args)
    if args.mode == "test":
        generate_test_data(args)