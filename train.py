from __future__ import division
from __future__ import print_function

import argparse
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.autograd import Variable

from model import GCNModelVAE, Discriminator,Discriminator_FC,Generator_FC
import itertools
from optimizer import loss_function
from utils import load_data,load_data2, mask_test_edges, preprocess_graph, get_roc_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=256, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--lambda2', type=float, default=1)
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')
#['cornell', 'texas', 'washington', 'wiscosin', 'uai2010'] 'cora','citeseer','pubmed'
args = parser.parse_args()
print(args)

# compute ac and nmi
def get_NMI(n_clusters,emb,true_label):
    from sklearn.cluster import KMeans
    from sklearn import metrics
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    predict_labels = kmeans.predict(emb)
    nmi = metrics.normalized_mutual_info_score(true_label, predict_labels)
    ac=clusteringAcc(true_label, predict_labels)
    return nmi,ac
def clusteringAcc(true_label, pred_label):
    from sklearn import metrics
    from munkres import Munkres, print_matrix
    # best mapping between true_label and predict label
    l1 = list(set(true_label))
    numclass1 = len(l1)
    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(true_label, new_predict)
    '''
    f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
    precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
    recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
    f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
    precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
    recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
    '''
    return acc
    
def log(x):
    return torch.log(x + 1e-8)
def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    #load dataset
    if args.dataset_str in ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010']:
        adj, features, label = load_data2(args.dataset_str)
    else:
        adj, features, label = load_data(args.dataset_str)
    #print(adj.shape,features.shape)
    print('adj类型:',type(adj))
    print('adj维度:', adj.shape)
    print('features类型:',type(features))
    print('features维度:', features.shape)

    #adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    #print(adj.shape)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    #adj = adj_train
    adj=adj_orig
    adj_train=adj
    #T=torch.FloatTensor(adj.todense())

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    mini_batch=adj.shape[0]
    print(mini_batch,pos_weight,norm,adj_norm.shape,adj_label.shape)
    #define model
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    print(model)
    G = Generator_FC(args.hidden2, args.hidden1, feat_dim)
    print(G)
    D = Discriminator_FC(args.hidden2, args.hidden1, feat_dim)
    print(D)
    D2 = Discriminator(feat_dim)
    print(D2)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    D2_optimizer = optim.Adam(D2.parameters(), lr=args.lr)
    #G_solver = torch.optim.Adam(itertools.chain(E.parameters(), G.parameters()), lr=lr, betas=[0.5,0.999], weight_decay=2.5*1e-5)
    #D_solver = torch.optim.Adam(D.parameters(), lr=lr, betas=[0.5,0.999], weight_decay=2.5*1e-5)

    hidden_emb = None
    NMI=[]
    AC=[]
    embedding=[]
    for epoch in range(args.epochs):
        d_loss=0
        g_loss=0
        cur_loss=0
        t = time.time()
        
        ####################
        model.train()
        optimizer.zero_grad()
        recovered, mu = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        ####################
        ''''''
        D2.train()
        D2_optimizer.zero_grad()
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mini_batch, args.hidden2))))
        X_hat = G(z)
        D_result = D2(X_hat)
        D_fake_loss= D_result
        D_result=D2(features)
        D_real_loss= D_result
        D_train_loss = -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))
        D_train_loss.backward(retain_graph=True)
        d_loss=D_train_loss.item()
        D2_optimizer.step()
        ##############
        D.train()
        D_optimizer.zero_grad()
        
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        
        #recovered, mu, logvar = model(features, adj_norm)
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mini_batch, args.hidden2))))
        X_hat = G(z)
        #loss=loss_function(preds=recovered, labels=adj_label, mu=mu, logvar=logvar, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)
        D_result = D(X_hat,z)
        #D_real_loss= D.loss(D_result,y_real_)
        #D_real_loss= torch.nn.ReLU()(1.0 - D_result).mean()
        D_fake_loss= D_result
        
        recovered, mu = model(features, adj_norm)
        D_result=D(features,mu)
        #loss=loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        #D_fake_loss= D.loss(D_result,y_fake_)
        #D_fake_loss= torch.nn.ReLU()(1.0 + D_result).mean()
        D_real_loss= D_result
        #D_loss = -torch.mean(log(D_enc) + log(1 - D_gen))
        #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss + loss
        #D_train_loss = 0.1*D_real_loss + 0.1*D_fake_loss
        D_train_loss = -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))
        #D_train_loss.backward(retain_graph=True)
        D_train_loss.backward(retain_graph=True)
        d_loss=D_train_loss.item()
        D_optimizer.step()
        
        
        #################
        model.train()
        G.train()
        D.eval()
        D2.eval()
        optimizer.zero_grad()
        optimizer_G.zero_grad()
        
        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (mini_batch, args.hidden2))))
        D_result = D(X_hat,z)
        D_fake_loss= D_result
        D_result = D2(X_hat)
        D2_fake_loss= D_result
        
        recovered, mu = model(features, adj_norm)
        loss=loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        D_result = D(features,mu)
        D_real_loss= D_result
        
        #G_train_loss= 0.1*D_fake_loss+args.lambda1*loss #+args.lambda2*torch.trace(mu.t().mm(T).mm(mu))
        #G_train_loss= -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))+args.lambda1*loss 
        G_train_loss= -0.01*torch.mean(log(D_real_loss))+args.lambda1*loss 
        G_train_loss.backward()
        optimizer.step()
        G2_train_loss= -torch.mean(log(D_fake_loss)+log(D2_fake_loss))
        G2_train_loss.backward()
        g_loss=G_train_loss.item()
        optimizer_G.step()
        ##################
        
        
        hidden_emb = mu.data.numpy()
        #print(hidden_emb)
        nmi,ac=get_NMI(int(label.max()+1),hidden_emb,label)
        NMI.append(nmi)
        AC.append(ac)
        embedding.append(hidden_emb)
        
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              g_loss,d_loss,'nmi:%.5f'%nmi,'ac:%.5f'%ac,
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    #roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    #print('Test ROC score: ' + str(roc_score))
    #print('Test AP score: ' + str(ap_score))
    
    print("Optimization Finished!")
    print(max(NMI),NMI.index(max(NMI)))
    print(max(AC),AC.index(max(AC)))
    import scipy.io as sio
    emb=embedding[NMI.index(max(NMI))]
    print(emb.shape)
    #sio.savemat(args.dataset_str+'-Embedding.mat', {"H": emb})
    sio.savemat(args.dataset_str+'-'+str(args.hidden2)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})
    #sio.savemat(args.dataset_str+'-lambda1-'+str(args.lambda1)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})
    #sio.savemat(args.dataset_str+'-lambda2-'+str(args.lambda2)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})


if __name__ == '__main__':
    #args.dataset_str='pubmed'
    gae_for(args)
    '''
    for i in ['cornell', 'texas', 'washington', 'wiscosin']:
        args.dataset_str=i
        gae_for(args)
    '''
    '''
    for i in ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010', 'cora','citeseer','pubmed']:
        args.dataset_str=i
        gae_for(args)
    '''
    '''
    for i in [0.1, 0.5, 1, 5, 10]:
        args.dataset_str='citeseer'
        args.lambda1=i
        gae_for(args)
    '''
    '''
    for i in [16, 32, 64, 128]:
        args.dataset_str='citeseer'
        args.hidden1=i*2
        args.hidden2=i
        gae_for(args)
    '''