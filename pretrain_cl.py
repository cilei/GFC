import math
import argparse
import time,os


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from util import *
from loader import PretrainDataset
from model import my_PretrainNN

def get_gnn_sim(z,centroids_z_list,dis_sum):
    a = torch.zeros((len(z),3),device=z.device)
    
    for index,i in enumerate(z):
        for jndex,j in enumerate(centroids_z_list):
            a[index,jndex] = torch.sum(torch.abs(i-j))

    a = a/dis_sum
    return a
     

def train(args,model,loader,optimizer,centroids_fp_list,mse_loss,centroids_z_list,dis_sum,logger):
    train_loss_all = 0
    model.train()

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(args.device)
        z = model(batch)
        pre = get_gnn_sim(z,centroids_z_list,dis_sum)
        label_true = smiliar_true(batch.fp,centroids_fp_list)
        loss = mse_loss(label_true,pre)
        train_loss_all += loss.item()
        loss.backward() 
        optimizer.step()
    logger.info(f"train loss:{(train_loss_all/len(loader)):.5}")
    
        
def load_args():
    parser = argparse.ArgumentParser()

    # seed & device
    parser.add_argument('--device_no', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, 
                        help="Seed for splitting the dataset.")
   
    #dataset
    parser.add_argument('--data_dir', type=str, default='./data/ZINC15', 
                        help='directory of pre-training data')
    parser.add_argument('--dataset', type=str, default='ZINC15', 
                        help='root directory of dataset')

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
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')


    #optimizer
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')



    ## clustering

    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--gate_dim', type=int, default=50, 
                        help="gate embedding space dimension, 50 or 300")

    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")


    return args

def main(args):
    
    set_seed(args.seed)
    
    logger = create_file_logger(os.path.join("log", 'log.txt'))
    logger.info(f"\n\n======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    logger.info("=======Setting=======")
    

    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info(f"{k}: {v}")

    # load data
    if not os.path.exists(args.data_dir):
        print("Data directory not found!")
        return

    train_set = PretrainDataset(root=args.data_dir,
                                mol_filename='zinc15_250k.txt')
    
    logger.info(f"train data num: {len(train_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = my_PretrainNN(args)

    model = model.to(args.device)

    logger.info(f"\n\n==get_z_smiles_fp start====={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")

    zs_init,fp_z,dis_sum = get_z_smiles_fp(model, train_loader, args.device,logger)

    logger.info(f"\n\n==get_max_distance finish====={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    
    centroids,centroids_index = init_centroid_index(model, zs_init, args.num_experts)

    logger.info(f"\n\n==centroids====={centroids}=======\n")

    logger.info(f"\n\n==init_centroid_index finish====={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")

    
    centroids_fp_list = []
    centroids_z_list = []
    for index in  centroids_index:
        
        centroids_fp_list.append(fp_z[index])
        centroids_z_list.append(zs_init[index])

    '''
    # lambda p: p.requires_grad 中，p是参数，而 p.requires_grad是返回值
    结合第2个函数所知 filter(lambda p: p.requires_grad, model.parameters())中 
    lambda p: p.requires_grad就是以p为参数的满足p.requires_grad的true的条件的函数。
    而参数p赋值的元素从列表model.parameters()里取。
    所以只取param.requires_grad = True（模型参数的可导性是true的元素），就过滤掉为false的元素。
    '''

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)

    mse_loss = nn.MSELoss()

    mse_loss.to(args.device)
    logger.info("\n=======Pre-train Start=======")

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1 :04d}")
        train(args,model,train_loader,optimizer,centroids_fp_list,mse_loss,centroids_z_list,dis_sum,logger)

    torch.save(model.state_dict(),args.output_model_file)
    logger.info(f"\n\n======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")


if __name__ == "__main__":
    
    args = load_args()
    main(args)