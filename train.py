import argparse
import imageio
import importlib
from tqdm import tqdm
import wandb
import os
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.utils import cal_metrics, create_dir, Logger, ExcelFile, LFdivide, LFintegrate, ycbcr2rgb
from dataset import TrainSetDataLoader, MultiTestSetDataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sr_model')
    parser.add_argument('--project', type=str, default="system_exp_4XSR")

    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

    parser.add_argument('--model_name', type=str, default='LF_InterNet_freq', help="model name")
    parser.add_argument('--data_name', type=str, default='ALL',
                        help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
    parser.add_argument('--path_for_train', type=str, default='./datasets/data_for_training/')
    parser.add_argument('--path_for_test', type=str, default='./datasets/data_for_test/')
    parser.add_argument('--path_log', type=str, default='./log/')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')
    parser.add_argument('--debug', nargs='?', const=True, default=False, help='debug mode')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=0, )

    args = parser.parse_args()
    return args


def train(train_loader, device, net, criterion, optimizer, args):
    ''' training one epoch '''
    psnr_iter_train = []
    loss_iter_train = []
    ssim_iter_train = []
    if not args.debug:
        wandb.watch(net)

    for idx_iter, (data, label, data_info) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        data = data.to(device)      # low resolution
        label = label.to(device)    # high resolution
        out = net(data, data_info)
        loss = criterion(out, label, data_info)

        optimizer.zero_grad()
        losses = 0
        loss_log = {}
        for key, values in loss.items():
            losses += values
            loss_log[key] = values.data.cpu()

        losses.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        loss_iter_train.append(losses.data.cpu())
        psnr, ssim = cal_metrics(args, label, out)
        psnr_iter_train.append(psnr)
        ssim_iter_train.append(ssim)
        loss_log.update({"loss":float(losses.data.cpu()), "train_psnr":psnr, "train_ssim":ssim})
        if not args.debug:
            wandb.log(loss_log)

    loss_epoch_train = float(np.array(loss_iter_train).mean())
    psnr_epoch_train = float(np.array(psnr_iter_train).mean())
    ssim_epoch_train = float(np.array(ssim_iter_train).mean())

    return loss_epoch_train, psnr_epoch_train, ssim_epoch_train


def main(args):
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    logger = Logger(log_dir, args)
    
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)

    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=True,pin_memory=True)

    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)

    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    net = MODEL.get_model(args)
    net.apply(MODEL.weights_init)
    start_epoch = 0
    net = net.to(device)
    cudnn.benchmark = True

    logger.log_string('PARAMETER ...')
    logger.log_string(args)

    criterion = MODEL.get_loss(args).to(device)

    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        loss_epoch_train, psnr_epoch_train, ssim_epoch_train = train(train_loader, device, net, criterion, optimizer, args)
        logger.log_string('The %dth Train, loss is: %.5f, psnr is %.5f, ssim is %.5f' %
                          (idx_epoch + 1, loss_epoch_train, psnr_epoch_train, ssim_epoch_train))

        save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
        args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch + 1)
        state = {
            'epoch': idx_epoch + 1,
            'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
        }
        torch.save(state, save_ckpt_path)
        logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))
        if idx_epoch>5:
            old_path = str(checkpoints_dir) + '/%s_%dx%d_%dx_epoch_%02d_model.pth' % (
                           args.model_name, args.angRes_in, args.angRes_in, args.scale_factor, idx_epoch-5)
            os.remove(old_path)

        with torch.no_grad():
            ''' Create Excel for PSNR/SSIM '''
            excel_file = ExcelFile()

            psnr_testset = []
            ssim_testset = []
            for index, test_name in enumerate(test_Names):
                test_loader = test_Loaders[index]

                epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                epoch_dir.mkdir(exist_ok=True)
                save_dir = epoch_dir.joinpath(test_name)
                save_dir.mkdir(exist_ok=True)

                psnr_iter_test, ssim_iter_test, LF_name = test(test_loader, device, net, save_dir)
                excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)

                psnr_epoch_test = float(np.array(psnr_iter_test).mean())
                ssim_epoch_test = float(np.array(ssim_iter_test).mean())

                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                logger.log_string('The %dth Test on %s, psnr/ssim is %.2f/%.3f' % (
                idx_epoch + 1, test_name, psnr_epoch_test, ssim_epoch_test))
            
            psnr_mean_test = float(np.array(psnr_testset).mean())
            ssim_mean_test = float(np.array(ssim_testset).mean())
            if not args.debug:
                wandb.log({"test_psnr":psnr_mean_test, "test_ssim":ssim_mean_test})

            logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.5f'
                                % (psnr_mean_test, ssim_mean_test))
            excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation.xls')

        ''' scheduler '''
        scheduler.step()


def test(test_loader, device, net, save_dir=None):
    LF_iter_test = []
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, data_info, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        [Lr_angRes_in, Lr_angRes_out] = data_info
        data_info[0] = Lr_angRes_in[0].item()
        data_info[1] = Lr_angRes_out[0].item()

        Lr_SAI_y = Lr_SAI_y.squeeze().to(device)  # numU, numV, h*angRes, w*angRes
        Hr_SAI_y = Hr_SAI_y
        Sr_SAI_cbcr = Sr_SAI_cbcr

        ''' Crop LFs into Patches '''
        subLFin = LFdivide(Lr_SAI_y, args.angRes_in, args.patch_size_for_test, args.stride_for_test)
        numU, numV, H, W = subLFin.size()
        subLFin = rearrange(subLFin, 'n1 n2 a1h a2w -> (n1 n2) 1 a1h a2w')
        subLFout = torch.zeros(numU * numV, 1, args.angRes_in * args.patch_size_for_test * args.scale_factor,
                               args.angRes_in * args.patch_size_for_test * args.scale_factor)

        ''' SR the Patches '''
        for i in range(0, numU * numV, args.minibatch_for_test):
            tmp = subLFin[i:min(i + args.minibatch_for_test, numU * numV), :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out = net(tmp.to(device), data_info)
                subLFout[i:min(i + args.minibatch_for_test, numU * numV), :, :, :] = out
        subLFout = rearrange(subLFout, '(n1 n2) 1 a1h a2w -> n1 n2 a1h a2w', n1=numU, n2=numV)

        ''' Restore the Patches to LFs '''
        Sr_4D_y = LFintegrate(subLFout, args.angRes_out, args.patch_size_for_test * args.scale_factor,
                              args.stride_for_test * args.scale_factor, Hr_SAI_y.size(-2)//args.angRes_out, Hr_SAI_y.size(-1)//args.angRes_out)
        Sr_SAI_y = rearrange(Sr_4D_y, 'a1 a2 h w -> 1 1 (a1 h) (a2 w)')

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim = cal_metrics(args, Hr_SAI_y, Sr_SAI_y)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        LF_iter_test.append(LF_name[0])


        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)
            Sr_SAI_ycbcr = torch.cat((Sr_SAI_y, Sr_SAI_cbcr), dim=1)
            Sr_SAI_rgb = (ycbcr2rgb(Sr_SAI_ycbcr.squeeze().permute(1, 2, 0).numpy()).clip(0,1)*255).astype('uint8')
            Sr_4D_rgb = rearrange(Sr_SAI_rgb, '(a1 h) (a2 w) c -> a1 a2 h w c', a1=args.angRes_out, a2=args.angRes_out)

            # save the SAI
            # path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
            # imageio.imwrite(path, Sr_SAI_rgb)
            # save the center view
            img = Sr_4D_rgb[args.angRes_out // 2, args.angRes_out // 2, :, :, :]
            path = str(save_dir_) + '/' + LF_name[0] + '_' + 'CenterView.bmp'
            imageio.imwrite(path, img)
            # save all views
            for i in range(args.angRes_out):
                for j in range(args.angRes_out):
                    img = Sr_4D_rgb[i, j, :, :, :]
                    path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
                    imageio.imwrite(path, img)

    return psnr_iter_test, ssim_iter_test, LF_iter_test


if __name__ == '__main__':
    args = get_args()
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 16
    args.task_name = args.model_name
    if not args.debug:
        wandb.init(project=args.project, name=args.model_name)
        config = wandb.config
        config.dropout = 0.01
    main(args)
