import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from p3d_model_synthetic import *
import torchvision
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from scipy.sparse import coo_matrix
from torch_geometric.nn import (SAGPooling, GraphConv, GCNConv, GATConv,
                                SAGEConv,TopKPooling)
import torch.nn.functional as F
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        #print(x.size())
        #print(edge_index.size())
        #print("in_channels:",self.in_channels)
        score = self.score_layer(x,edge_index).squeeze()


        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm
"""
There are several versions of GCN layer implemetation
"""
#Type1: from GNN Benchmark
#Type2: used in MIPR
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    Without batch normalization and residual connection
    """
    def __init__(self, in_features, out_features,bias=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = bias
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model,num_AU, num_classes,  t=0, adj_file=None):
        
        super(GCNResnet, self).__init__()
        # For r3d, i.e.,3D cnn, output shape is (512,1)
        #self.features = nn.Sequential(
         #   model.stem,
          #  model.layer1,
           # model.layer2,
           # model.layer3,
            #model.layer4,
            #model.avgpool,
        #)
        
        # For p3d
        self.features = P3D199(pretrained=False,num_classes=18)
   
        ##################
        #self.num_classes = num_classes
        self.num_AU = num_AU
        #self.pooling = nn.MaxPool2d(14, 14)


        #self.gc1 = GraphConvolution(num_AU, 512) #1024
        #self.gc1_ = GraphConvolution(1, 512) 
        #self.gc2 = GraphConvolution(1024, 512)# origin
        #self.gc2 = GraphConvolution(512, 19) #for p3d
        #self.gc2_ = GraphConvolution(512, 1)
        #self.relu = nn.LeakyReLU(0.2)
        #self.dropout = nn.Dropout(0.3)
        #self.fc1 = nn.Linear(19, 15)
        #self.fc2 = nn.Linear(15, num_classes)
        #_adj = gen_A(num_AU, t, adj_file)
        import pickle
        self.adj_file = pickle.load(open(adj_file, 'rb'), encoding='utf-8')
        #print(self.adj_file)
        #assert 0
        #self.A = Parameter(torch.from_numpy(adj_file).float()) #(19,19)
        #self.graph_pool = SAGPooling(in_channels=1, ratio=0.5, GNN=GraphConv)   

       
        #####Adding SAGPool###########
        self.in_channels = 1 #x:（19，1）
        self.nhid = 32
        self.num_classes = 7#args.num_classes
        self.pooling_ratio = 0.5 # 0.5
        self.dropout_ratio = 0.5
   
        self.conv1 = GCNConv(in_channels = self.in_channels, out_channels = self.nhid)
        self.score_layer = GCNConv(self.nhid,1)        
        self.pool1 = SAGPool(self.nhid*3, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        #self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        #self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = nn.Linear(self.nhid*3*2*2, self.nhid) #(self.nhid*4, self.nhid)
        self.lin2 = nn.Linear(self.nhid, self.nhid//2) #(self.nhid, self.nhid//2)
        self.lin3 = nn.Linear(self.nhid//2, 3)     #(self.nhid//2, self.num_classes)  

    def forward(self, feature, inp):
        feature = self.features(feature) # as input to gcn
        #print("p3d model output size:",feature.size())#torch.Size([1, 19])!!
        feature_ = feature.transpose(0, 1)
        #print("p3d model output size after transpose:",feature_.size())#torch.Size([19,1])!!
    
         #gcn
        #adj = gen_adj(self.A).detach() #(19,19)

        # COO format
        A_coo = coo_matrix(self.adj_file)
        #print(A_coo)
        #print(A_coo.row)
        #print(A_coo.col)
        a = []
        a.append(A_coo.row)
        a.append(A_coo.col)
        adj = np.asarray(a)
        #print(adj)
 
        edge_index = torch.from_numpy(adj).long().cuda()

        # 3 GCN layers
        gcn1 = F.relu(self.conv1(feature_,edge_index))
        #print(edge_index.size())
        #print(gcn1.size())
        #assert 0
        gcn2 = F.relu(self.conv2(gcn1,edge_index))
        gcn3 = F.relu(self.conv3(gcn2,edge_index))   
        gcn_feature = torch.cat((gcn1,gcn2,gcn3), dim=1)
        #print("gcn_feature:",gcn_feature.size()) #(19,384)
        # batch setting
        #batch = np.arange(19)
        batch = np.ones((18))
        batch = torch.from_numpy(batch).long().cuda()
        #SAGPooling layer
        x, edge_index, _, batch, _ = self.pool1(gcn_feature, edge_index, None, batch)
        #print("pool1:",x.size())#(10,384)

        #readout layer
        readout = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        #print("readout:",readout.size()) #(2,768)
        x = readout.view(1,-1)# convert (2,768) to (1,-1)
        fc1 = F.relu(self.lin1(x)) 
        #print("fc1:",fc1.size())#(2,128)
        #x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        fc2 = F.relu(self.lin2(fc1))

        x = F.log_softmax(self.lin3(fc2), dim=-1)

        
        return x, feature

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.conv1.parameters(), 'lr': lr},
                {'params': self.conv2.parameters(), 'lr': lr},
                {'params': self.conv3.parameters(), 'lr': lr},
                {'params': self.pool1.parameters(), 'lr': lr},
                
                {'params': self.score_layer.parameters(), 'lr': lr},
                {'params': self.lin1.parameters(), 'lr': lr},
                {'params': self.lin2.parameters(), 'lr': lr},
                {'params': self.lin3.parameters(), 'lr': lr},

                ]



def gcn_resnet3d(num_classes,  t, adj_file, num_AU=18, pretrained=True):
    #model = models.resnet101(pretrained=pretrained)
    #model = models.video.r3d_18(pretrained=pretrained)
    model = models.video.r3d_18(pretrained=False)
    model = P3D199(pretrained=False,num_classes=num_classes)
 
    return GCNResnet(model, num_classes=num_classes, num_AU=num_AU, t=t ,adj_file=adj_file)


def resnet3d(num_classes,  pretrained=True):
    model = models.video.r3d_18(pretrained=False)
    #model = p3d()
    return merResnet(model, num_classes=num_classes)
