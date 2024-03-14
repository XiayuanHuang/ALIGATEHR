#################################################################
# Code written by Xiayuan Huang (xiayuan.huang@yale.edu)
# For bug issues, please contact author using the email address
#################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from pedigree import EPedigreesAttentionLayer
from ontology import Ontology
from readData import ReadData 


class ALIGATEHR(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(ALIGATEHR, self).__init__()
        self.dropout = dropout

        self.attentions = [EPedigreesAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = EPedigreesAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
