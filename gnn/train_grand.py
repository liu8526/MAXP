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
from pytorch_toolbelt import losses as L

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
    batch_feature = node_feats[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_feature, batch_labels

def gpu_train(GPUid, DATA_PATH,
              gnn_model, hidden_dim, n_layers, n_classes, fanouts,
              batch_size=32, num_workers=4, epochs=100, lr=0.003, 
              output_folder='./output'):
    if not os.path.exists(output_folder): os.makedirs(output_folder)

    device_id = "cuda:" + GPUid
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
    graph = graph.int()
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    graph.ndata['feature'] = node_feat
    graph.ndata['labels'] = labels

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    sampler = MultiLayerNeighborSampler(fanouts)
    train_dataloader = NodeDataLoader(graph,
                                      train_nid,
                                      sampler,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=num_workers,
                                      )
    val_dataloader = NodeDataLoader(graph,
                                    val_nid,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    in_feat = node_feat.shape[1]
    from model_grand import GRAND
    sample = 4
    lam = 1.0
    tem = 0.5
    model = GRAND(in_feat, 
                  hidden_dim, 
                  n_classes, 
                  sample, #sample
                  8,#order
                  0.2, #dropnode_rate
                  0.2, #input_droprate
                  0.2, #hidden_droprate
                 True)#use_bn
    def consis_loss(logps, temp, lam):
        ps = [th.exp(p) for p in logps]
        ps = th.stack(ps, dim = 2)
        
        avg_p = th.mean(ps, dim = 2)
        sharp_p = (th.pow(avg_p, 1./temp) / th.sum(th.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()

        sharp_p = sharp_p.unsqueeze(2)
        loss = th.mean(th.sum(th.pow(ps - sharp_p, 2), dim = 1, keepdim=True))

        loss = lam * loss
        return loss

    model = model.to(device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss().to(device_id)
    # loss_fn = L.JointLoss(L.FocalLoss(), thnn.CrossEntropyLoss(), 0.7, 0.3).to(device_id)
    # loss_fn = L.FocalLoss().to(device_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # ------------------- 4. Train model  ----------------------------------------------- #
    print('Plan to train {} epoches \n'.format(epochs))
    # model.load_state_dict(th.load("../save/graphattn-Epoch_0.pth"))
    for epoch in range(epochs):
        # mini-batch for training
        train_loss_list = []
        train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # batch_feature, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
            # blocks = [block.to(device_id) for block in blocks]
            with th.no_grad():
                c = th.Tensor([]).to(th.int32)
                for block in blocks:
                    c=th.cat([c, block.srcdata[dgl.NID]],0)
                s_graph = dgl.node_subgraph(graph, c).to(device_id)
                batch_labels = s_graph.ndata['labels']
                label_mask = batch_labels!=-1
                batch_labels = batch_labels[label_mask]
        

            loss_sup = 0
            train_pred_logits = model(s_graph, s_graph.ndata['feature'], True)            
            # calculate supervised loss
            for k in range(sample):
                loss_sup += F.nll_loss(train_pred_logits[k][label_mask], batch_labels)            
            loss_sup = loss_sup/sample            
            # calculate consistency loss
            loss_consis = consis_loss(train_pred_logits, tem, lam)            
            train_loss = loss_sup + loss_consis

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.cpu().detach().numpy())
            tr_batch_pred = th.sum(th.argmax(train_pred_logits[0][label_mask], dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])
            train_acc_list.append(tr_batch_pred.cpu().detach().numpy())

            if step % 100 == 0 :
                print('In epoch:{:03d}|step {:03} of {:03d} , train_loss:{:.4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                train_dataloader.dataloader.dataset.shape[0]//train_dataloader.dataloader.batch_size,
                                                                                                np.mean(train_loss_list),
                                                                                                np.mean(train_acc_list)))
                train_loss_list = []
                train_acc_list = []
            if step % 5000 == 0 and step>0:
                model_path = os.path.join(output_folder, gnn_model + '-Epoch_' + str(epoch)+ '.pth')
                model_para_dict = model.state_dict()
                th.save(model_para_dict, model_path)
        if epoch%1 == 0:
            model_path = os.path.join(output_folder, gnn_model + '-Epoch_' + str(epoch)+ '.pth')
            model_para_dict = model.state_dict()
            th.save(model_para_dict, model_path)

        # mini-batch for validation
        val_loss_list = []
        val_acc_list = []
        model.eval()
        with th.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                # forward
                # batch_feature, batch_labels = load_subtensor(node_feat, labels, seeds, input_nodes, device_id)
                # blocks = [block.to(device_id) for block in blocks]

                c = th.Tensor([]).to(th.int32)
                for block in blocks:
                    c=th.cat([c, block.srcdata[dgl.NID]],0)
                s_graph = dgl.node_subgraph(graph, c).to(device_id)
                batch_labels = s_graph.ndata['labels']
                label_mask = batch_labels!=-1
                batch_labels = batch_labels[label_mask]

                # metric and loss
                val_pred_logits = model(s_graph, s_graph.ndata['feature'], False)                  
                val_loss = F.nll_loss(val_pred_logits[label_mask], batch_labels)            

                val_loss_list.append(val_loss.detach().cpu().numpy())
                val_batch_pred = th.sum(th.argmax(val_pred_logits[label_mask], dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])
                val_acc_list.append(val_batch_pred.detach().cpu().numpy())
            print('Val In epoch:{:03d}| val_loss:{:.4f}, val_acc:{:.4f}'.format(epoch,
                                                                                np.mean(val_loss_list),
                                                                                np.mean(val_acc_list)))
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
    parser.add_argument('--hidden_dim', type=int, required=False, default=128)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument("--fanout", type=str, required=False, help="fanout numbers", default='0')
    parser.add_argument('--batch_size', type=int, required=False, default=1000)
    parser.add_argument('--GPU', type=str, required=True, default=0)
    parser.add_argument('--num_workers_per_gpu', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
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
    GPUid = args.GPU
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
    print('GPU list: {}'.format(GPUid))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))
    print('Output path: {}'.format(OUT_PATH))

    gpu_train(GPUid, DATA_PATH,
                gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=23,
                fanouts=FANOUTS, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS, lr=LR, 
                output_folder=OUT_PATH)