from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from ALIGATEHR import ALIGATEHR

# Training settings
parser = argparse.ArgumentParser()  
parser.add_argument('seq_file', type=str, metavar='<visit_file>', help='The path to the Pickled file containing visit information of patients')
parser.add_argument('label_file', type=str, metavar='<label_file>', help='The path to the Pickled file containing label information of patients')
parser.add_argument('tree_file', type=str, metavar='<tree_file>', help='The path to the Pickled files containing the ancestor information of the input medical codes. Only use the prefix and exclude ".level#.pk".')
parser.add_argument('out_file', metavar='<out_file>', help='The path to the output models. The models will be saved after every epoch')
parser.add_argument('--embed_file', type=str, default='', help='The path to the Pickled file containing the representation vectors of medical codes. If you are not using medical code representations, do not use this option')
parser.add_argument('--embed_size', type=int, default=128, help='The dimension size of the visit embedding. If you are providing your own medical code vectors, this value will be automatically decided. (default value: 128)')
parser.add_argument('--rnn_size', type=int, default=128, help='The dimension size of the hidden layer of the GRU (default value: 128)')
parser.add_argument('--attention_size', type=int, default=128, help='The dimension size of hidden layer of the MLP that generates the attention weights (default value: 128)')
parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
parser.add_argument('--n_epochs', type=int, default=100, help='The number of training epochs (default value: 100)')
parser.add_argument('--L2', type=float, default=0.001, help='L2 regularization coefficient for all weights except RNN (default value: 0.001)')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate used for the hidden layer of RNN (default value: 0.5)')
parser.add_argument('--log_eps', type=float, default=1e-8, help='A small value to prevent log(0) (default value: 1e-8)')
parser.add_argument('--verbose', action='store_true', help='Print output after every 100 mini-batches (default false)')
args = parser.parse_args()

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer

model = ALIGATEHR(nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()

def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
