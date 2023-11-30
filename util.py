import logging
import random
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import pairwise_distances_argmin
from rdkit import DataStructs


class Scf_index:
    def __init__(self, dataset, args):
        self.device = args.device
        
        self.max_scf_idx = None
        self.scfIdx_to_label = None
        self.num_scf = None

        self.get_scf_idx(dataset)


    def get_scf_idx(self, dataset):
        ''''
        scf label: scf 에 속한 sample 수가 많은 순서부터 desending order 로 sorting 해서 label 매김
        self.num_train_scf = train set에 있는 scf 의 종류 
        self. 
        '''
        
        scf = defaultdict(int)
        max_scf_idx = 0 
        for data in dataset:
            idx = data.scf_idx.item()
            scf[idx] += 1
            if max_scf_idx < idx:
                max_scf_idx = idx
        self.max_scf_idx = max_scf_idx
        scf = sorted(scf.items(), key=lambda x: x[1], reverse=True)
        
        self.scfIdx_to_label = torch.ones(max_scf_idx+1).to(torch.long).to(torch.long) * -1
        self.scfIdx_to_label = self.scfIdx_to_label.to(self.device) 

        for i, k in enumerate(scf):
            self.scfIdx_to_label[k[0]] = i 

        self.num_scf = len(scf)


def load_models(args, model):
    
    if not args.ckpt_all == "":

        load = torch.load(args.ckpt_all)
        mis_keys, unexp_keys = model.load_state_dict(load, strict=False)
        print('missing_keys:', mis_keys)
        print('unexpected_keys:', unexp_keys)
    
    elif not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)



### utils for eval
def cal_roc(y_true, y_scores):
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
    
    return sum(roc_list) / (len(roc_list) + 1e-10) * 100



def init_centroid_index(model, z_s, num_experts):
    
    z_s_arr = z_s.detach().cpu().numpy()

    num_data = z_s_arr.shape[0]

    if  num_data> 35000:
        mask_idx = list(range(num_data))
        random.shuffle(mask_idx)
        z_s_arr = z_s_arr[mask_idx[:35000]]

    kmeans = KMeans(n_clusters=num_experts, random_state=0).fit(z_s_arr)
    
    centroids = kmeans.cluster_centers_

    k_means_centroidsIndex = pairwise_distances_argmin(centroids, z_s_arr)
            
    # 画图
    '''
    tsne = TSNE(n_components=2).fit_transform(z_s_arr)
    aa = tsne[:, 0]
    bb = tsne[:, 1]
    color = ['limegreen', 'cornflowerblue', 'orange']
    plt.figure(dpi=300)
    for i,label in enumerate(kmeans.labels_):
        plt.scatter(aa[i], bb[i], facecolor=color[label], alpha=0.7)
    plt.savefig('./tsne_4.jpg')
    '''

    return centroids,k_means_centroidsIndex

    # model.cluster.data = torch.tensor(centroids).to(model.cluster.device)

def get_z_smiles_fp(model, loader, device,logger):
    model.train()
    
    z_s = [] 
    fp_s = []
       
    dis_ = []
    for i,batch in enumerate(loader):
        batch = batch.to(device)
        
        with torch.no_grad():
            z= model(batch)
        
        for x,index in enumerate(z):
            for j,jndex in enumerate(z):
                dis_.append(torch.sum(torch.abs(index-jndex)).item())
        
        z_s.append(z)
        fp_s.extend(batch.fp)
    
    max_distance = max(dis_)

    z_s = torch.cat(z_s, dim=0)
    
    return z_s,fp_s,max_distance



def get_max_distance(model,loader,device):

    model.train()
    
    dis_ = []
    for i,batch in enumerate(loader):
        batch = batch.to(device)
        
        with torch.no_grad():
            z= model(batch)
        
        for i,index in enumerate(z):
            for j,jndex in enumerate(z):
                dis_.append(torch.sum(torch.abs(index-jndex)))

    max_distance = dis_[0]
    for dis in dis_:
        if dis.item() > max_distance.item():
            max_distance = dis
    
    return max_distance


    
def smiliar_true(batch_fp,centroids_fp_list):
    a = torch.zeros((len(batch_fp),3),device="cuda:1")
    for i,fp in enumerate(batch_fp):
        for j,cenFp in enumerate(centroids_fp_list):
            a[i,j] = DataStructs.FingerprintSimilarity(fp,cenFp)
    
    return a    
    

    



def set_seed(seed):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger