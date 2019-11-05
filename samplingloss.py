from collections import defaultdict

import torch
import torch.nn as nn
from torch import multinomial

import torch.nn.functional as F
import numpy as np

from torch.distributions.categorical import Categorical

class SamplingLoss(nn.Module):
    '''SamplingLoss calculates an approximate softmax'''
    def __init__(self, hidden_size, k, obj, Z, noise, q, b, g):
        super(SamplingLoss, self).__init__()

        self.hidden_size = hidden_size

        self.obj = obj
        
        self.k = k
        self.noise = Categorical(torch.cuda.FloatTensor(noise))

        self.q = q
        self.b = b
        self.g = g
        
        self.Z = Z
        if Z == 'learn' and obj == 'discri':
            self.Z = nn.Parameter(torch.cuda.FloatTensor([len(noise)]).log())

            
    def forward(self, weight, bias, hiddens, targets, training=True):
        total_loss = None
        if len(hiddens.size()) > 2: hiddens = hiddens.view(-1, hiddens.size(2))
        
        if training: 
            samples = multinomial(torch.cuda.FloatTensor(self.noise.probs), self.k, replacement=True)
            
            target_weight = weight[targets]
            target_bias = bias[targets]
            target_model = torch.sum(torch.mul(hiddens, target_weight), dim=1).unsqueeze(1) + target_bias.unsqueeze(1)
        
            noise_weight = weight[samples]
            noise_bias = bias[samples]
            noise_model = torch.nn.functional.linear(hiddens, noise_weight, bias=noise_bias)
                
            target_noise = self.noise.logits[targets].unsqueeze(-1)
            noise_noise = self.noise.logits[samples]
          
            model_logits = torch.cat([target_model, noise_model], dim=1)
            noise_logits = torch.cat([target_noise, noise_noise.repeat(targets.size(0),1) ],dim=1)

            output = model_logits - noise_logits
            new_targets = torch.zeros_like(targets)
                        
            if self.obj == 'discri':
                """ Discriminative objective: we minimize a divergence between binary probabilites for 
                the right word and the noise samples. In this particular case, we can choose to ignore
                the partition function, hence the possibility to use a supplementary context-independant
                parameter to replace it - or we can re-weight the noise distribution with the model
                distribution, in order to make the partition function disappear (as done in blackOut)
                """
                if self.Z == 'learn':
                    output = output - self.Z
                elif self.Z == 'blackout': # Assuming here that outputs contains ratios ! 
                    noise_offset = torch.logsumexp(output[:,1:], dim=1).unsqueeze(1)
                    output = output - noise_offset
                if not self.Z == 'blackout':
                    output -= torch.cuda.FloatTensor([self.k]).log()
                
                labels = torch.zeros_like(output).cuda().scatter_(1, new_targets.unsqueeze(1), 1.0)
                if not (self.g == 1.0):
                    res_pos = F.logsigmoid(output)
                    res_neg = F.logsigmoid( - output)
                    loss = - res_pos[:,0] - ( 1.0 / (self.g - 1.0) ) * logsumexp((self.g - 1.0) * res_neg[:,1:], dim=1)
                    loss_b = (1.0 / self.g) * (logsumexp(self.g*res_pos, dim=1) + logsumexp(self.g*res_neg, dim=1))
                    total_loss = (loss + loss_b).squeeze().sum()
                elif not (self.q == 1.0):
                    res_pos = (1.0 - self.q) * F.logsigmoid(output[:,0])
                    res_neg = (1.0 - self.q) * F.logsigmoid( - output[:,1:])
                    loss = res_pos.exp() + res_neg.exp().sum(dim=1)
                    total_loss = ( - 1.0 / ( self.q * (1.0 - self.q)) ) * loss.squeeze().sum()
                elif not (self.b == 1.0):
                    res_pos =  F.logsigmoid(output)
                    res_neg =  F.logsigmoid( - output)
                    loss = - (1.0 / (self.b - 1.0))*(((self.b - 1.0)*res_pos[:,0]).exp()+((self.b - 1.0)*res_neg[:,1:]).exp().sum(dim=1))
                    loss_b = (1.0 / self.b) * ( (self.b*res_pos).exp().sum(dim=1) + (self.b*res_neg).exp().sum(dim=1) )
                    total_loss = (loss + loss_b).squeeze().sum()
                else:
                    total_loss =  F.binary_cross_entropy_with_logits(output, 
                                                                     labels,
                                                                     reduction='sum')                

            elif self.obj == 'importance':
                """ We apply the divergence in the sub-space of probability distributions, but we choose
                to approximate the partition function using self-importance sampling - it has the main
                effect of having the sum over the noise samples un-normalized probabilities happen inside
                the log function, allowing the use of a log_softmax over the k+1 words
                """
                if not (self.g == 1.0):
                    res = self.log_g(output)
                elif not (self.q == 1.0):
                    res = self.log_q(output)
                elif not (self.b == 1.0):
                    res = self.log_b(output)
                else:
                    res = F.log_softmax(output, dim=-1)
                total_loss = -torch.gather(res, dim=1, index=new_targets.view(-1, 1)).squeeze().float().sum()

            else:
                """ In this case, we apply directly the divergence on positive measures, approximating the
                sum over the vocabulary using 'simple' importance sampling """
                if not (self.g == 1.0):
                    res = - target_model + (1.0 / self.g) * logsumexp(self.g * output, dim=1)
                elif  not (self.q == 1.0):
                    res = (1.0 / (self.q * (1.0 - self.q))) * ( self.q * torch.sum( output.exp(), dim=1).unsqueeze(1) - (self.q * target_model).exp() )
                elif not (self.b == 1.0):
                    res =  (- 1.0 / (self.b - 1.0)) * ((self.b - 1.0) * target_model).exp() + (1.0 / self.b) * (self.b * output).exp().sum(dim = 1)
                else:
                    res = torch.sum(output.exp(), dim=1).unsqueeze(1) - target_model
                total_loss = res.squeeze().sum()

            return (total_loss / len(targets)).type_as(weight)
                
        else:
            output = F.linear(hiddens, weight, bias=bias)
            total_loss = -torch.gather(F.log_softmax(output, dim=-1), dim=1, index=targets.view(-1, 1)).squeeze().float().sum()
            return (total_loss / len(targets)).type_as(weight)

    def log_b(self, logits):
        lnorm = logsumexp(logits, dim=-1, keepdim=True)
        lsum = torch.exp(self.b * logits).sum(dim=1, keepdim=True) * torch.exp(- self.b * lnorm)
        return torch.exp((self.b - 1.0)*(logits - lnorm)) / (self.b - 1.0)  - lsum / self.b - 1.0 / ((self.b - 1.0) * self.b)

    def log_g(self, logits):
        lsum = torch.exp(self.g * logits).sum(dim=1, keepdim=True)
        return logits - torch.log(lsum) / self.g

    def log_q(self, logits):
        lnorm = logsumexp(logits, dim=-1, keepdim=True)
        return torch.expm1((1.0 - self.q) * (logits - lnorm)) / ((1.0 - self.q) * self.q)
    
def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)
                                                                                                            
def log_q(logits, q):
    res = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return torch.expm1((1.0 - q) * res) / ((1.0 - q) * q)

def log_b(logits, b):
    res = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    tot = ( 1.0 / b) * torch.exp(b * res) - ( 1.0 / (1.0 + b)) * torch.exp((1+b) * res).sum(dim=1, keepdim=True)
    return tot
            
