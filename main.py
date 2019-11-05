import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data
import model
from torch.autograd import Variable

from utils import batchify, get_batch, repackage_hidden

import gc

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--log-file', type=str,  default='',
                    help='path to save the log')

# Modification for Sampling Loss
parser.add_argument('--sampling_loss', action='store_true',
                    help='use approximated loss using sampling')
parser.add_argument('--k', type=int, default=1024,
                    help='in case of sampling loss, number of samples by batch')
parser.add_argument('--obj', default='default',
                    help='How to apply the divergence to obtain the objective: discriminatively (discri), ranking (importance), or directly on unormalized probabilities (default)')
parser.add_argument('--Z', default='default',
                    help='In case of discriminative objective, how to deal with the partition function: (learn) will use one supplementary parameter to learn a context independant substitute; (blackout) will modify the noise distribution to remove the need for computing Z')

# Modification for specific output bias initialization
parser.add_argument('--bias_log_freq', action='store_true',
                    help='use the log frequency of words as output bias - useful with discriminative sampling objectives')

# Modification for distortion:
parser.add_argument('--q', type=float, default=1.0,
                    help='Entropy distortion = alpha divergence')
parser.add_argument('--b', type=float, default=1.0,
                    help='Beta divergence')
parser.add_argument('--g', type=float, default=1.0,
                    help='Gamma divergence')

args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
print('Producing dataset...')
corpus = data.Corpus(args.data)

# Modification for sampling
if args.bias_log_freq or args.sampling_loss:
    unigram = [corpus.dictionary.counter[w] for w in corpus.dictionary.idx2word]

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

print(train_data.size(0))

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
from samplingloss import SamplingLoss
criterion = None
ntokens = len(corpus.dictionary)

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
# Modification for output bias initialization
if args.bias_log_freq:
    model.init_output_bias(unigram)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if (not criterion):
    if args.sampling_loss:
        criterion = SamplingLoss(args.emsize, k=args.k, obj=args.obj, Z=args.Z, noise=unigram, q=args.q, b=args.b, g=args.g)  
    else:
        criterion = SplitCrossEntropyLoss(args.emsize, q=args.q, b=args.b, g=args.g)
###

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)                
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets, training=False)[0].data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def train(epoch):
    inner_product = 0
    count = 0
    save_hiddens = []
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        
        # Activation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.data

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            if (args.sampling_loss) and (args.obj is not 'importance'):
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | bpc {:8.3f}'.format(
                          epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                          elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            else:
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                          epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                          elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
finetune = False
# At any point you can hit Ctrl + C to break out of training early.

try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    # For evaluation before training - necessary for saved models
    if args.resume:
        val_ce2 = evaluate(val_data)
        print('-' * 89)
        print('valid ce {:5.2f} | '
              'valid ppl {:8.2f} '.format(
                  val_ce2, math.exp(val_ce2)))
        print('-' * 89)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(epoch)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for name, prm in model.named_parameters():
                if not 'module.weight_hh_l0' in name:
                    tmp[prm] = prm.data.detach().clone()
                    prm.data = optimizer.state[prm]['ax'].data.detach().clone()

            val_loss2 = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ce {:5.2f} | '
                  'valid ppl {:8.2f}'.format(
                      epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2)))
            print('-' * 89)
            
            if epoch % 30 == 0:
                test_loss = evaluate(test_data, test_batch_size)
                print('| end of epoch {:3d} | time: {:5.2f}s | test ce {:5.2f} | '
                      'test ppl {:8.2f} | test bpc {:8.3f}'.format(
                          epoch, (time.time() - epoch_start_time), test_loss, math.exp(test_loss)))
                print('=' * 89)
                                        
            if val_loss2 < stored_loss: # or (args.sampling_loss and args.discri and not args.blackout):              
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for name, prm in model.named_parameters():
                if not 'module.weight_hh_l0' in name:
                    prm.data = tmp[prm].data.detach().clone()

            if (not finetune and epoch == args.epochs/2) or ((not finetune and (len(best_val_loss)>args.nonmono and val_loss2 > min(best_val_loss[:-args.nonmono])))):
                finetune = True
                print('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            best_val_loss.append(val_loss2)
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid ce {:5.2f} | '
                  'valid ppl {:8.2f}'.format(
                      epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < stored_loss: # or (args.sampling_loss and args.discri and not args.blackout):
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if epoch % 30 == 0:
                test_loss = evaluate(test_data, test_batch_size)
                print('| end of epoch {:3d} | time: {:5.2f}s | test ce {:5.2f} | '
                      'test ppl {:8.2f} | test bpc {:8.3f}'.format(
                          epoch, (time.time() - epoch_start_time), test_loss, math.exp(test_loss)))
                print('=' * 89)

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
        
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
if not args.resume:
    model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

