import os
from utils.utility import set_seed
from utils import dataloader, earlystopping, utility
import argparse
import configparser
import logging
import torch
import pandas as pd
from model import models



def get_parameters():
    parser = argparse.ArgumentParser(description='dynGNN for traffic prediction')
    parser.add_argument('--enable_cuda', type=bool, default='True',
                        help='enable CUDA, default as True')
    parser.add_argument('--n_pred', type=int, default=3,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--epochs', type=int, default=2,
                        help='epochs, default as 500')
    parser.add_argument('--dataset_config_path', type=str, default='./config/data/train/road_traffic/metr-la.ini',
                        help='the path of dataset config file, (1) pemsd7-m.ini for PeMSD7-M, \
                            (2) metr-la.ini for METR-LA, and (3) pems-bay.ini for PEMS-BAY')
    parser.add_argument('--model_config_path', type=str, default='./config/model/gcnconv_sym_glu.ini',
                        help='the path of model config file, \
                            (1) gcnconv_sym_glu.ini for STGCN(GCNConv, Kt=3)')
    parser.add_argument('--opt', type=str, default='AdamW',
                        help='optimizer, default as AdamW')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    config = configparser.ConfigParser()

    def ConfigSectionMap(section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
                if dict1[option] == -1:
                    logging.debug("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_config_path = args.model_config_path
    dataset_config_path = args.dataset_config_path

    config.read(dataset_config_path, encoding="utf-8")

    #read the dataset_config_file in the dataset_config_path
    dataset = ConfigSectionMap('data')['dataset']
    time_intvl = int(ConfigSectionMap('data')['time_intvl'])
    n_his = int(ConfigSectionMap('data')['n_his'])
    Kt = int(ConfigSectionMap('data')['kt'])
    stblock_num = int(ConfigSectionMap('data')['stblock_num'])
    if ((Kt - 1) * 2 * stblock_num > n_his) or ((Kt - 1) * 2 * stblock_num <= 0):
        raise ValueError(f'ERROR: {Kt} and {stblock_num} are unacceptable.')
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    drop_rate = float(ConfigSectionMap('data')['drop_rate'])
    batch_size = int(ConfigSectionMap('data')['batch_size'])
    learning_rate = float(ConfigSectionMap('data')['learning_rate'])
    weight_decay_rate = float(ConfigSectionMap('data')['weight_decay_rate'])
    step_size = int(ConfigSectionMap('data')['step_size'])
    gamma = float(ConfigSectionMap('data')['gamma'])
    data_path = ConfigSectionMap('data')['data_path']
    wam_path = ConfigSectionMap('data')['wam_path']
    model_save_path = ConfigSectionMap('data')['model_save_path']

    config.read(model_config_path, encoding="utf-8")

    # read the model_config_file in the model_config_path
    gated_act_func = ConfigSectionMap('casualconv')['gated_act_func']

    graph_conv_type = ConfigSectionMap('graphconv')['graph_conv_type']
    if (graph_conv_type != "chebconv") and (graph_conv_type != "gcnconv"):
        raise NotImplementedError(f'ERROR: {graph_conv_type} is not implemented.')
    else:
        graph_conv_type = graph_conv_type

    Ks = int(ConfigSectionMap('graphconv')['ks'])
    if (graph_conv_type == 'gcnconv') and (Ks != 2):
        Ks = 2

    mat_type = ConfigSectionMap('graphconv')['mat_type']

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    day_slot = int(24 * 60 / time_intvl)
    n_pred = args.n_pred

    time_pred = n_pred * time_intvl
    time_pred_str = str(time_pred) + '_mins'
    model_name = ConfigSectionMap('graphconv')['model_name']
    model_save_path = model_save_path + model_name + '_' + dataset + '_' + time_pred_str + '.pth'

    adj_mat = dataloader.load_weighted_adjacency_matrix(wam_path)

    n_vertex_vel = pd.read_csv(data_path, header=None).shape[1]
    n_vertex_adj = pd.read_csv(wam_path, header=None).shape[1]
    if n_vertex_vel != n_vertex_adj:
        raise ValueError(
            f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
    else:
        n_vertex = n_vertex_vel

    opt = args.opt
    epochs = args.epochs

    # make sure model and data are transferred to GPU device if detected
    if graph_conv_type == "chebconv":
        if (mat_type != "wid_sym_normd_lap_mat") and (mat_type != "wid_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {args.mat_type} is wrong.')
        mat = utility.calculate_laplacian_matrix(adj_mat, mat_type)
        chebconv_matrix = torch.from_numpy(mat).float().to(device)
        stgcn_chebconv = models.STGCN_ChebConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type,
                                               chebconv_matrix, drop_rate).to(device)
        model = stgcn_chebconv

    elif graph_conv_type == "gcnconv":
        if (mat_type != "hat_sym_normd_lap_mat") and (mat_type != "hat_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {args.mat_type} is wrong.')
        mat = utility.calculate_laplacian_matrix(adj_mat, mat_type)
        gcnconv_matrix = torch.from_numpy(mat).float().to(device)
        stgcn_gcnconv = models.STGCN_GCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type,
                                             gcnconv_matrix, drop_rate).to(device)
        model = stgcn_gcnconv

    return device, n_his, n_pred, day_slot, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma








if __name__ == "__main__":
    seed = 1688825600
    set_seed(seed)

    device, n_his, n_pred, day_slot, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma = get_parameters()


