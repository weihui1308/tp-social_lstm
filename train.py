import argparse
import logging
import sys
import torch
import os
import numpy as np
from loss import displacement_error, final_displacement_error

from loader import data_loader
from utils import get_dset_path, get_mean_error, checkpoint_path
from model import LSTM_model


parser = argparse.ArgumentParser()

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--shuffle', default=True)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')

# Model Options
parser.add_argument('--input_dim', default=2, type=int)
parser.add_argument('--output_dim', default=2, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--rnn_dim', default=128, type=int)
parser.add_argument('--infer', default=False)

# Misc
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def main(args):
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    # 随机种子
    # torch.manual_seed(2)
    # np.random.seed(2)
    # if args.use_gpu:
    #     torch.cuda.manual_seed_all(2)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    log_path = './log/'
    log_file_curve = open(os.path.join(log_path, 'log_loss.txt'), 'w+')
    log_file_curve_val = open(os.path.join(log_path, 'log_loss_val.txt'), 'w+')
    log_file_curve_val_ade = open(os.path.join(log_path, 'log_loss_val_ade.txt'), 'w+')

    net = LSTM_model(args)
    if args.use_gpu:
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    #接着上次训练的地方继续训练
    # restore_path = '.\model\lstm294.tar'
    # logger.info('Restoring from checkpoint {}'.format(restore_path))
    # checkpoint = torch.load(restore_path)
    # net.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #
    # for i_epoch in range(checkpoint['epoch']+1):
    #     if (i_epoch + 1) % 100 == 0:
    #         args.learning_rate *= 0.98

    epoch_loss_min = 160
    epoch_smallest = 0
    #for epoch in range(checkpoint['epoch']+1, args.num_epochs):
    for epoch in range(args.num_epochs):
        count = 0
        batch_loss = 0

        for batch in train_loader:
            # Zero out gradients
            net.zero_grad()
            optimizer.zero_grad()

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            num_ped = obs_traj.size(1)
            pred_traj_gt = pred_traj_gt

            #model_teacher.py
            pred_traj = net(obs_traj, num_ped, pred_traj_gt, seq_start_end)
            loss = displacement_error(pred_traj, pred_traj_gt)
            #loss = get_mean_error(pred_traj, pred_traj_gt)

            # Compute gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            # Update parameters
            optimizer.step()

            batch_loss += loss
            count += 1

            #print(loss / num_ped)
        if (epoch + 1) % 6 == 0:
            pass
            #scheduler.step()
        logger.info('epoch {} train loss is {}'.format(epoch, batch_loss/count))
        log_file_curve.write(str(batch_loss.item()/count) + "\n")

        batch_loss = 0
        val_ade = 0
        total_ade = 0
        for idx, batch in enumerate(val_loader):
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            num_ped = obs_traj.size(1)
            pred_traj_gt = pred_traj_gt

            # model_teacher.py
            pred_traj = net(obs_traj, num_ped, pred_traj_gt, seq_start_end)
            loss = displacement_error(pred_traj, pred_traj_gt)

            batch_loss += loss
            val_ade += loss / (num_ped * 12)
            total_ade += val_ade

            count += 1

        fin_ade = total_ade / (idx + 1)
        log_file_curve_val_ade.write(str(fin_ade.item()) + "\n")

        epoch_loss = batch_loss / count
        if epoch_loss_min > epoch_loss:
            epoch_loss_min = epoch_loss
            epoch_smallest = epoch

            logger.info('Saving model')
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch))
        logger.info('epoch {} val loss is {}'.format(epoch, epoch_loss))
        log_file_curve_val.write(str(epoch_loss.item()) + "\n")
        logger.info('epoch {} is smallest loss is {}'.format(epoch_smallest, epoch_loss_min))
        logger.info('the smallest ade is {}'.format(total_ade/(idx+1)))
        logger.info("-"*50)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)