from collections import defaultdict

import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

class SplitCrossEntropyLoss(nn.Module):
    r'''SplitCrossEntropyLoss calculates an approximate softmax'''
    def __init__(self, hidden_size, q, b, g):
        super(SplitCrossEntropyLoss, self).__init__()
        self.hidden_size = hidden_size
        # Distortion of the softmax / entropy
        self.q = q
        self.b = b
        self.g = g

    def forward(self, weight, bias, hiddens, targets, training=True):
        total_loss = None
        if len(hiddens.size()) > 2: hiddens = hiddens.view(-1, hiddens.size(2))
        all_head_res = torch.nn.functional.linear(hiddens, weight, bias = bias)

        # Modification for distortion        
        if ((not (self.q == 1.0)) and training):
            softmaxed_all_head_res, sum_res = self.log_q(all_head_res)
        elif ((not (self.b == 1.0)) and training):
            softmaxed_all_head_res, sum_res = self.log_b(all_head_res)
        elif ((not (self.g == 1.0)) and training):
            softmaxed_all_head_res, sum_res = self.log_g(all_head_res)
        else:
            softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)
            
        running_offset = 0
        total_loss = -torch.gather(softmaxed_all_head_res, dim=1, index=targets.view(-1, 1)).float().sum()
        return (total_loss / len(targets)).type_as(weight)

    def log_b(self, logits):
        lnorm = logsumexp(logits, dim=-1, keepdim=True)
        lsum = torch.exp(self.b * logits).sum(dim=1, keepdim=True) * torch.exp(- self.b * lnorm)
        return torch.exp((self.b - 1.0)*(logits - lnorm)) / (self.b - 1.0)  - lsum / self.b - 1.0 / ((self.b - 1.0) * self.b), lsum 

    def log_g(self, logits):
        lnorm = logsumexp(logits, dim=-1, keepdim=True)
        lsum = torch.exp(self.g * logits).sum(dim=1, keepdim=True) 
        return logits - torch.log(lsum) / self.g, lsum * torch.exp(- self.g * lnorm)

    def log_q(self, logits):
        lnorm = logsumexp(logits, dim=-1, keepdim=True)
        return torch.expm1((1.0 - self.q) * (logits - lnorm)) / ((1.0 - self.q) * self.q), torch.exp(lnorm)

def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)
