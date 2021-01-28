import torch
import torch.nn as nn
from torch.autograd import Variable


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class LSTM_model(nn.Module):
    def __init__(self, args):
        super(LSTM_model, self).__init__()

        self.args = args

        self.embedding_dim = args.embedding_dim
        self.input_dim = args.input_dim
        self.num_layers = args.num_layers
        self.rnn_dim = args.rnn_dim
        self.output_dim = args.output_dim
        self.seq_len = args.obs_len + args.pred_len
        self.pred_len = args.pred_len
        self.obs_len = args.obs_len
        self.use_gpu = args.use_gpu
        self.infer = args.infer

        self.input_layer = nn.Linear(self.input_dim, self.embedding_dim)
        self.lstm_layer = nn.LSTM(2, self.rnn_dim, self.num_layers)
        # self.lstm_layer = nn.LSTMCell(2, self.rnn_dim)

        self.output_layer = nn.Linear(self.rnn_dim, self.output_dim)

        self.pool_net = SocialPooling(
            h_dim=self.rnn_dim,
            activation='relu',
            batch_norm=True,
            dropout=0.0,
            neighborhood_size=2.0,
            grid_size=8
        )

    def forward(self, obs_traj, num_ped, pred_traj_gt, seq_start_end):
        pred_traj = Variable(torch.zeros(self.pred_len, num_ped, 2))  # tode

        output, (hn, cn) = self.lstm_layer(obs_traj[0].unsqueeze(0))
        #pred_traj[0] = obs_traj[-1] + self.output_layer(hn)  # gt[0]

        for j in range(1, self.obs_len):
            end_pos = obs_traj[j]
            pool_h = self.pool_net(hn, seq_start_end, end_pos).unsqueeze(0)
            output, (hn, cn) = self.lstm_layer(obs_traj[j].unsqueeze(0), (pool_h, cn))
            # output, (hn, cn) = self.lstm_layer(obs_traj[j + 1].view(1, num_ped, 2), (hn, cn))
        pred_traj[0] = obs_traj[-1] + self.output_layer(hn)
        for i in range(1, self.pred_len):
            if self.infer:
                end_pos = pred_traj[i - 1]
                pool_h = self.pool_net(hn, seq_start_end, end_pos).unsqueeze(0)
                output, (hn, cn) = self.lstm_layer(pred_traj[i - 1].unsqueeze(0), (pool_h, cn))
                traj_tmp = self.output_layer(hn)
                pred_traj[i] = end_pos + traj_tmp
            else:
                end_pos = pred_traj_gt[i - 1]
                pool_h = self.pool_net(hn, seq_start_end, end_pos).unsqueeze(0)
                output, (hn, cn) = self.lstm_layer(pred_traj_gt[i - 1].unsqueeze(0), (pool_h, cn))
                traj_tmp = self.output_layer(hn)
                pred_traj[i] = end_pos + traj_tmp

        return pred_traj


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""

    def __init__(
            self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
            neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start  # 4
            grid_size = self.grid_size * self.grid_size  # 64
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]  # (4,128)
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)  # (16,128)
            curr_end_pos = end_pos[start:end]  # (4,2)
            curr_pool_h_size = (num_ped * grid_size) + 1  # 257
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))  # (257,128)
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)  # (4,2)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)  # (16,2)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)  # (16,2)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                top_left, curr_end_pos).type_as(seq_start_end)  # 16
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))  # 16
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound  # 16
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size  # 64
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)  # 4
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)  # (16,128)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]  # (257,128)
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h
