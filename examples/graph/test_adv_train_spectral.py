import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import SpectralAttack, PGDAttack
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
from deeprobust.graph.data import PtbDataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--reg_weight', type=float, default=5.0,  help='regularization weight')
parser.add_argument('--loss_type', type=str, default='CE', choices=['CE', 'CW'], help='loss type')
parser.add_argument('--adversary', type=str, default='Spectral', choices=['PGD', 'Spectral'], help='model variant')
parser.add_argument('--device', type=str, default='cuda:0', help='model variant')

args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print (device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='gcn')
adj, features, labels = data.adj, data.features, data.labels
# features = normalize_feature(features)

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Target Model
model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, weight_decay=5e-4, device=device)
model = model.to(device)

print("Data {} | Loss type {} | Perturbation Rate {} | Reg weight {:.4f}".format(
      args.dataset, args.loss_type, args.ptb_rate, args.reg_weight))

# test on original adj
print('=== test on original adj ===')
model.fit(features, adj, labels, idx_train)
output = model.output
acc_test = accuracy(output[idx_test], labels[idx_test])
print("Test set results:",
      "accuracy= {:.4f}".format(acc_test.item()))

##################### Adversary #####################
adv_train_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, weight_decay=5e-4, device=device)
adv_train_model = adv_train_model.to(device)
adv_train_model.initialize()

print('=== Adversarial Training for Evasion Attack===')
if args.adversary == 'Spectral':
    adversary = SpectralAttack(model=adv_train_model, 
                              nnodes=adj.shape[0], 
                              loss_type=args.loss_type, 
                              regularization_weight=args.reg_weight,
                              device=device)
else:
    adversary = PGDAttack(model=adv_train_model, nnodes=adj.shape[0], loss_type='CE', device=device)

adversary = adversary.to(device)

perturbations = int(args.ptb_rate * (adj.sum()//2))
for i in tqdm(range(20)):
    # modified_adj = adversary.attack(features, adj)
    adversary.attack(features, adj, labels, idx_train, perturbations, epochs=20)
    modified_adj = adversary.modified_adj
    adv_train_model.fit(features, modified_adj, labels, idx_train, train_iters=50, initialize=False)

adv_train_model.eval()
# test directly or fine tune
print('=== test on clean adj ===')
acc_clean = adv_train_model.test_reset_graph(idx_test, features=features, adj=adj)

print('=== test on SpectralAttack adj ===')
attack1 = SpectralAttack(model=adv_train_model, 
                              nnodes=adj.shape[0], 
                              loss_type=args.loss_type, 
                              regularization_weight=args.reg_weight,
                              device=device)
attack1 = attack1.to(device)
attack1.attack(features, adj, labels, idx_test, perturbations, epochs=20)
modified_adj = attack1.modified_adj
acc_spectral_ptb = adv_train_model.test_reset_graph(idx_test, features=features, adj=modified_adj)

print('=== test on TopologyAttack adj ===')
attack2 = PGDAttack(model=adv_train_model, nnodes=adj.shape[0], loss_type='CE', device=device)
attack2 = attack2.to(device)
attack2.attack(features, adj, labels, idx_test, perturbations, epochs=20)
modified_adj = attack2.modified_adj
acc_pgd_ptb = adv_train_model.test_reset_graph(idx_test, features=features, adj=modified_adj)

print("Data {} | Loss type {} | Perturbation Rate {} | Reg weight {:.4f}".format(
      args.dataset, args.loss_type, args.ptb_rate, args.reg_weight))
print("Clean Acc {:.4f} | Spectral Acc {:4f} | Pgd Acc {:.4f}".format(
      acc_clean, acc_spectral_ptb, acc_pgd_ptb))

