import numpy as np

import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import GATConv
from torch_scatter import scatter_add,scatter_sum
import math



num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3



class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) ## modify

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        # x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        x = self.weight_linear(x).view(-1, self.heads * self.emb_dim)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):

        x_i = x_i.view(-1, self.heads, self.emb_dim) # nodes x edges x heads x emb
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.heads * self.emb_dim )

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim).mean(dim=1)
        # aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self._initialize_weights()

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            elif gnn_type == 'gat1':
                self.gnns.append(GATConv(emb_dim, heads=1))


        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if JK == 'w_sum':
            initial_scalar_parameters = [0.0] * (num_layer + 1)
            self.para = ParameterList(
                [
                    Parameter(
                        torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=True
                    )
                    for i in range(num_layer + 1)
                ])
        
        self.q = nn.Linear(6,1)
        self.k = nn.Linear(6,1)
        self.v = nn.Linear(6,1)
        self.scale = 6 ** -0.5

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr,batch = argv[0], argv[1], argv[2],argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

   
        
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
    

        h_list = [x]
        for layer in range(self.num_layer):
            
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
           
            h = self.batch_norms[layer](h)
           
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        h_list_new = []
        
        for i in (h_list):
            i = scatter_sum(i,batch,dim=0,dim_size=batch[-1]+1)
            h_list_new.append(i)
        
        t = torch.empty((len(h_list_new),batch[-1]+1,300),device="cuda:1")
        for i,list in enumerate(h_list_new):
            t[i]=list
        
        batch_x = torch.empty((batch[-1]+1,300),device="cuda:1")

        for x in range(batch[-1]+1):
            
            q = self.q(t[:,x,:].reshape(300,6))
            k = self.k(t[:,x,:].reshape(300,6))
            
            v = self.v(t[:,x,:].reshape(300,6))
            
            attn = (q @ k.transpose(-2,-1)) * self.scale
            attn = attn.softmax(dim=-1)
            x_ = attn @ v
            batch_x[x,:] = x_.reshape(1,300)
     
        ### Different implementations of Jk-concat
        # if self.JK == "concat":
        #     node_representation = torch.cat(h_list, dim=1)
        # elif self.JK == "last":
        #     node_representation = h_list[-1]
        # elif self.JK == "max":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)
        # elif self.JK == "sum":
        #     h_list = [h.unsqueeze_(0) for h in h_list]
        #     node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        # else:
        #     raise ValueError("unmatched argument.")
     

        return batch_x

    def _initialize_weights(self):
    # https://github.com/facebookresearch/deepcluster/blob/main/models/alexnet.py
    # nn.Embedding is initialized by xavier_uniform_ from the original gnn code    
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class gate(torch.nn.Module):  
    def __init__(self, emb_dim, gate_dim=300):
        super(gate, self).__init__()
        self.linear1 = nn.Linear(emb_dim, gate_dim)
        self.batchnorm = nn.BatchNorm1d(gate_dim)
        self.linear2 = nn.Linear(gate_dim, gate_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        gate_emb = self.linear2(x)
        return gate_emb 

class my_PretrainNN(torch.nn.Module):
    def __init__(self, args):
        super(my_PretrainNN, self).__init__()
        self.num_layer = args.num_layer
        self.drop_ratio = args.dropout_ratio
        self.JK = args.JK
        self.emb_dim = args.emb_dim
        # self.num_tasks = args.num_tasks
        self.graph_pooling = args.graph_pooling
        self.gnn_type = args.gnn_type

        self.gate_pool = global_add_pool

        self.gate = gate(args.emb_dim, args.gate_dim)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(self.num_layer, self.emb_dim, self.JK, self.drop_ratio, gnn_type=self.gnn_type)

        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * self.emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(self.emb_dim, 1))
        elif self.graph_pooling[:-1] == "set2set":
            set2set_iter = int(self.graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * self.emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(self.emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_rep = self.gnn(x, edge_index, edge_attr,data.batch)
        
        
        # gate_input = self.gate_pool(node_rep, batch)
        

        z = self.gate(node_rep)
       

        return z
    

class GNN_finetune(torch.nn.Module):
    def __init__(self, args):
        super(GNN_finetune, self).__init__()

        self.gate_dim = args.gate_dim
        self.num_tasks = args.num_tasks
        
        self.pretrainNN = my_PretrainNN(args)

        self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.gate_dim, (self.gate_dim)//2),
            torch.nn.ELU(),
            torch.nn.Linear((self.gate_dim)//2, self.num_tasks)
        )
    
    def from_pretrained(self,model_file):
        
        self.pretrainNN.load_state_dict(torch.load(model_file))

    def forward(self, data):
       
        node_representation = self.pretrainNN(data)
        

        return self.graph_pred_linear(node_representation)
