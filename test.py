import argparse
import logging
import sys
import torch
import os
import numpy as np
from loader import data_loader
from utils import get_dset_path, get_mean_error
from model import LSTM_model
from loss import final_displacement_error

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
parser.add_argument('--shuffle', default=False)

# Optimization
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')
parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')

# Model Options
parser.add_argument('--input_dim', default=2, type=int)
parser.add_argument('--output_dim', default=2, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--rnn_dim', default=128, type=int)
parser.add_argument('--infer', default=True)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device_ids = [0, 1]
    test_path = get_dset_path(args.dataset_name, 'test')

    logger.info("Initializing test dataset")
    test_dset, test_loader = data_loader(args, test_path)

    net = LSTM_model(args)
    #net = net.cuda(device_ids[1])

    checkpoint_path = "./model/lstm348.tar"
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    count = 0
    total_ade = 0
    total_fde = 0
    for batch in test_loader:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch
        num_ped = obs_traj.size(1)   # (8 n 2)
        #pred_traj_gt = pred_traj_gt.cuda(device_ids[1])
        pred_traj = net(obs_traj, num_ped, pred_traj_gt, seq_start_end)
        ade = get_mean_error(pred_traj, pred_traj_gt)
        total_ade += ade
        fde = final_displacement_error(pred_traj[-1], pred_traj_gt[-1])
        total_fde += (fde / num_ped)
        #logger.info("ade is {:.2f}".format(ade))
        count += 1

    ade_fin = total_ade / count
    fde_fin = total_fde / count
    logger.info("ade is {:.2f}".format(ade_fin))
    logger.info("fde is {:.2f}".format(fde_fin))

def sava_traj(batch, idx_ped, pred_traj, pred_traj_gt, obs_traj):
    obs_traj = obs_traj.detach().numpy().reshape(args.obs_len, 2)
    pred_traj = pred_traj.cpu().detach().numpy().reshape(args.pred_len, 2)
    pred_traj_gt = pred_traj_gt.cpu().detach().numpy().reshape(args.pred_len, 2)
    pred_traj = np.concatenate((obs_traj, pred_traj), axis=0)
    pred_traj_gt = np.concatenate((obs_traj, pred_traj_gt), axis=0)
    fname = 'traj/'+str(batch)+'_'+str(idx_ped)+'_'+'pred_traj.txt'
    fname_gt = 'traj/'+str(batch) + '_' + str(idx_ped) + '_' + 'pred_traj_gt.txt'
    np.savetxt(fname, pred_traj)
    np.savetxt(fname_gt, pred_traj_gt)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)