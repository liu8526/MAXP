#-*- coding:utf-8 -*-

# Author:james Zhang
"""
    Minibatch training with node neighbor sampling in multiple GPUs
"""

import os
import argparse
import datetime as dt
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler

import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader

from models import GraphSageModel, GraphConvModel, GraphAttnModel
from utils import load_dgl_graph, time_diff
from model_utils import early_stopper, thread_wrapped_func


def load_subtensor(node_feats, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def cleanup():
    dist.destroy_process_group()


def gpu_train(proc_id, n_gpus, GPUS, DATA_PATH,
              gnn_model, hidden_dim, n_layers, n_classes, fanouts,
              batch_size=32, num_workers=4, epochs=100, lr=0.003, 
              message_queue=None,
              output_folder='./output'):

    device_id = GPUS[proc_id]
    print('Use GPU {} for training ......'.format(device_id))

    # ------------------- 1. Prepare data and split for multiple GPUs ------------------- #
    start_t = dt.datetime.now()
    print('Start graph building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))
                                                           
    # Retrieve preprocessed data and add reverse edge and self-loop
    graph, labels, train_nid, val_nid, test_nid, node_feat = load_dgl_graph(DATA_PATH)
    # graph = graph.int()
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    train_div, _ = divmod(train_nid.shape[0], n_gpus)
    val_div, _ = divmod(val_nid.shape[0], n_gpus)
    test_div, _ = divmod(test_nid.shape[0], n_gpus)

    # just use one GPU, give all training/validation index to the one GPU
    if proc_id == (n_gpus - 1):
        train_nid_per_gpu = train_nid[proc_id * train_div: ]
        val_nid_per_gpu = val_nid[proc_id * val_div: ]
        test_nid_per_gpu = test_nid[proc_id * test_div: ]
    # in case of multiple GPUs, split training/validation index to different GPUs
    else:
        train_nid_per_gpu = train_nid[proc_id * train_div: (proc_id + 1) * train_div]
        val_nid_per_gpu = val_nid[proc_id * val_div: (proc_id + 1) * val_div]
        test_nid_per_gpu = test_nid[proc_id * test_div: (proc_id + 1) * test_div]

    sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid_per_gpu,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      )
    val_dataloader = NodeDataLoader(graph,
                                    val_nid_per_gpu,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )

    test_dataloader = NodeDataLoader(graph,
                                    test_nid_per_gpu,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )

    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    start_t = dt.datetime.now()
    print('Start Model building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    in_feat = node_feat.shape[1]
    if gnn_model == 'graphsage':
        model = GraphSageModel(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = GraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0)
    elif gnn_model == 'graphattn':
        model = GraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([3] * n_layers), activation=F.relu, feat_drop=0, attn_drop=0)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')
    model = model.to(device_id)

    if n_gpus > 1:
        model = thnn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device_id],
                                                      output_device=device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    earlystoper = early_stopper(patience=2, verbose=False)

    # ------------------- 4. Train model  ----------------------------------------------- #
    print('Plan to train {} epoches \n'.format(epochs))
    for epoch in range(epochs):
        # mini-batch for training
        train_loss_list = []
        # train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # forward
            optimizer.zero_grad()
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            train_batch_logits = model(blocks, batch_inputs)
            train_loss = loss_fn(train_batch_logits, batch_labels)
            # backward
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.cpu().detach().numpy())
            tr_batch_pred = th.sum(th.argmax(train_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            if step % 100 == 0:
                print('In epoch:{:03d}|step {:04d} of {:04d} , train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                        step,
                                                                                                        train_dataloader.dataloader.dataset.shape[0]//train_dataloader.dataloader.batch_size,
                                                                                                        np.mean(train_loss_list),
                                                                                                        tr_batch_pred.detach()))

        # mini-batch for validation
        val_loss_list = []
        val_acc_list = []
        model.eval()
        for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
            # forward
            batch_inputs, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            val_batch_logits = model(blocks, batch_inputs)
            val_loss = loss_fn(val_batch_logits, batch_labels)

            val_loss_list.append(val_loss.detach().cpu().numpy())
            val_batch_pred = th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            if step % 10 == 0:
                print('In epoch:{:03d}|step {:04d} of {:04d}, val_loss:{:4f}, val_acc:{:.4f}'.format(epoch,
                                                                                            step,
                                                                                            val_dataloader.dataloader.dataset.shape[0]//val_dataloader.dataloader.batch_size,
                                                                                            np.mean(val_loss_list),
                                                                                            val_batch_pred.detach()))

        if epoch%5 == 0 and epoch>3:
            model_path = os.path.join(output_folder, gnn_model + '-Epoch_' + str(epoch)+ '.pth')
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)

    # -------------------------6. Save models --------------------------------------#
    model_path = os.path.join(output_folder, gnn_model + '-Epoch_' + str(epoch)+ '.pth')
    model_para_dict = model.state_dict()
    th.save(model_para_dict, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.", 
                        default='../../MAXPdata')
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn'],
                        required=False, default='graphsage')
    parser.add_argument('--hidden_dim', type=int, required=False, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument("--fanout", type=str, required=False, help="fanout numbers", default='30,30,30,30')
    parser.add_argument('--batch_size', type=int, required=False, default=1)#300000
    parser.add_argument('--GPU', nargs='+', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--out_path', type=str, required=False, help="path for saving model parameters",
                        default='../save')
    args = parser.parse_args()

    # parse arguments
    DATA_PATH = args.data_path
    MODEL_CHOICE = args.gnn_model
    HID_DIM = args.hidden_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    BATCH_SIZE = args.batch_size
    GPUS = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs
    LR = args.lr
    OUT_PATH = args.out_path

    # output arguments for logging
    print('Data path: {}'.format(DATA_PATH))
    print('Used algorithm: {}'.format(MODEL_CHOICE))
    print('Hidden dimensions: {}'.format(HID_DIM))
    print('number of hidden layers: {}'.format(N_LAYERS))
    print('Fanout list: {}'.format(FANOUTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('GPU list: {}'.format(GPUS))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))
    print('Output path: {}'.format(OUT_PATH))

    n_gpus = len(GPUS)
    gpu_train(0, n_gpus, GPUS, DATA_PATH,
                gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=23,
                fanouts=FANOUTS, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS, lr=LR, 
                message_queue=None, output_folder=OUT_PATH)