import torch
from deeprobust.graph.data import Dataset, PtbDataset
from deeprobust.graph.defense import GCN, GCNJaccard
import numpy as np
np.random.seed(15)

# load clean graph
data = Dataset(root='/tmp/', name='cora', setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

# load pre-attacked graph by mettack
perturbed_data = PtbDataset(root='/tmp/', name='cora')
perturbed_adj = perturbed_data.adj

# Set up defense model and test performance
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
model = model.to(device)
model.fit(features, perturbed_adj, labels, idx_train)
model.eval()
output = model.test(idx_test)

# Test on GCN
model = GCN(nfeat=features.shape[1], nclass=labels.max()+1, nhid=16, device=device)
model = model.to(device)
model.fit(features, perturbed_adj, labels, idx_train)
model.eval()
output = model.test(idx_test)