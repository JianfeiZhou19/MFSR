import importlib
import torch
import argparse
import torch.backends.cudnn as cudnn
from utils.utils import *
from dataset.datasets import MultiTestSetDataLoader
from collections import OrderedDict
from train import test

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sr_model')

    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

    parser.add_argument('--model_name', type=str, default='attention_rcab', help="model name")
    parser.add_argument('--data_name', type=str, default='ALL',
                        help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
    parser.add_argument('--path_for_train', type=str, default='./datasets/data_for_training/')
    parser.add_argument('--path_for_test', type=str, default='./datasets/data_for_test/')
    parser.add_argument('--path_log', type=str, default='./log/')
    parser.add_argument('--path_pre_pth', type=str, default='./log/SR_5x5_4x/MFSR/MFSR/checkpoints/MFSR_5x5_4x_epoch_50_model.pth')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')

    args = parser.parse_args()
    return args

def main(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.exp.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()

        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
            excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

            psnr_epoch_test = float(np.array(psnr_iter_test).mean())
            ssim_epoch_test = float(np.array(ssim_iter_test).mean())
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print('Test on %s, psnr/ssim is %.2f/%.3f' % (test_name, psnr_epoch_test, ssim_epoch_test))
            pass

        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        excel_file.add_sheet('ALL', 'Average', psnr_mean_test, ssim_mean_test)
        print('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean_test, ssim_mean_test))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')

    pass


if __name__ == '__main__':
    args = get_args()
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 16
    args.task_name = args.model_name
    args.use_pre_ckpt = True
    main(args)
