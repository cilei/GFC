import argparse
from cmath import inf

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import os,time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import tqdm
import numpy as np
from util import *


from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from splitters import scaffold_split, random_split
import pandas as pd
from model import GNN_finetune


criterion = nn.BCEWithLogitsLoss(reduction = "none")


def train(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        
        pred = model(batch)
        
        y = batch.y.view(pred.shape).to(torch.float64)
        
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)

        

        #loss matrix after removing null target 数据中可能有些空值，这一步就是把空值去掉
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
       
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()
       


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)


        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    #Whether y is non-null or not.
    y = batch.y.view(pred.shape).to(torch.float64)
    is_valid = y**2 > 0
    #Loss matrix
    loss_mat = criterion(pred.double(), (y+1)/2)
    #loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
    loss = torch.sum(loss_mat)/torch.sum(is_valid)


    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    eval_roc = sum(roc_list)/len(roc_list) #y_true.shape[1]

    return eval_roc, loss
    

def load_args():
    parser = argparse.ArgumentParser()

    # seed & device
    parser.add_argument('--device_no', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, 
                        help="Seed for splitting the dataset.")
   
    #dataset
    parser.add_argument('--data_dir', type=str, default='./dataset/sider', 
                        help='directory of finetune data')
    parser.add_argument('--dataset', type=str, default='sider', 
                        help='[bbbp, bace, muv, clintox, sider,tox21, toxcast,hiv]')

    parser.add_argument('--split', type=str, default="scaffold", 
                        help="random or scaffold or random_scaffold")

    #model
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
    parser.add_argument('--gnn_type', type=str, default="gin")


    # train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')


    #optimizer

    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--lr_pred', type=float, default=1e-3,
                        help='learning rate for the prediction layer (default: 0.0005)')
    parser.add_argument('--lr_feat', type=float, default=5e-4,
                        help='learning rate (default: 0.001)')


    ## clustering

    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--gate_dim', type=int, default=50, 
                        help="gate embedding space dimension, 50 or 300")

    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    
    parser.add_argument('--input_model_file', type=str, default = '../saved_model/pretrain.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--GNN_para', type=bool, default = True, help='if the parameter of pretrain update')

    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")


    return args


def main(args):

    logger = create_file_logger(os.path.join("log", 'log.txt'))
    logger.info(f"\n\n======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    logger.info("=======Setting=======")



    # Bunch of classification tasks
    if args.dataset == "tox21":
        args.num_tasks = 12
        args.num_classes = 2
    elif args.dataset == "hiv":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "muv":
        args.num_tasks = 17
        args.num_classes = 2
    elif args.dataset == "bace":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "bbbp":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "toxcast":
        args.num_tasks = 617
        args.num_classes = 2
    elif args.dataset == "sider":
        args.num_tasks = 27
        args.num_classes = 2
    elif args.dataset == "clintox":
        args.num_tasks = 2
        args.num_classes = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    else:
        raise ValueError("Invalid split option.")
    


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = GNN_finetune(args)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model.to(args.device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    
    if args.GNN_para:
        print('GNN update')
        model_param_group.append({"params": model.pretrainNN.parameters(), "lr":args.lr_feat})
    else:
        print('No GNN update')

    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr_pred})

    optimizer = optim.Adam(model_param_group, weight_decay=args.decay) 

   

    # optimizer = optim.Adam(model_param_group, weight_decay=args.decay)

    finetune_model_save_path = './model_checkpoints/' + args.dataset + '.pth'

    train_auc_list, test_auc_list = [], []

    for epoch in range(1, args.epochs+1):
        print('====epoch:',epoch)
            
        train(model, args.device, train_loader, optimizer)

        print('====Evaluation')
        
        train_auc, train_loss = eval(args, model, args.device, train_loader)

        val_auc, val_loss = eval(args, model, args.device, val_loader)
        test_auc, test_loss = eval(args, model, args.device, test_loader)

        test_auc_list.append(float('{:.4f}'.format(test_auc)))
        train_auc_list.append(float('{:.4f}'.format(train_auc)))

        torch.save(model.state_dict(), finetune_model_save_path)

            
        print("train_auc: %f val_auc: %f test_auc: %f" %(train_auc, val_auc, test_auc))

    logger.info("train_auc:")
    logger.info(train_auc_list)
    logger.info("test_auc:")
    logger.info(test_auc_list)
    logger.info(max(test_auc_list))
    logger.info(model)

    pass

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = load_args()
    main(args)
