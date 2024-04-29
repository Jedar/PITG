# Author: Laura Kulowski
import operator
import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import heapq

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb

    def eval(self):
        return self.logp
    
    def __lt__(self, node):
        return self.logp < node.logp

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 3, embed_size=200, dropout=0.3):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings = input_size, embedding_dim = self.hidden_size)
        self.lstm = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, num_layers = num_layers, batch_first=True)
    
    def forward(self, x_input, encoder_hidden=None, x_length=None):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        out = self.embedding(x_input)
        _, self.hidden = self.lstm(out)
        return self.hidden
    
    def init_hidden(self, batch_size, device=torch.device('cpu')):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class LSTMDecoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, input_size, decode_size, hidden_size, num_layers = 3, embed_size=128, dropout=0.3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.vocab_size = decode_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings = decode_size, embedding_dim = self.hidden_size)
        self.lstm = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, decode_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
        '''
        output = self.embedding(x_input)
        lstm_out, hidden = self.lstm(output, encoder_hidden_states)
        output = self.linear(lstm_out)
        output = self.softmax(output)
        return output, hidden
    
    def init_start_token(self, batch_size, start_token, device=torch.device('cpu')):
        return torch.tensor(
            [[start_token] for _ in range(batch_size)], dtype=torch.long).to(device)

class EntityEncoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, input_size, hidden_size, num_layers = 3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(EntityEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding1 = nn.Embedding(num_embeddings = input_size, embedding_dim = self.hidden_size*num_layers)
        self.embedding2 = nn.Embedding(num_embeddings = input_size, embedding_dim = self.hidden_size*num_layers)

    def forward(self, x_input):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
        '''
        hc = self.embedding1(x_input).view(self.num_layers, -1, self.hidden_size)
        hx = self.embedding2(x_input).view(self.num_layers, -1, self.hidden_size)
        return (hc, hx)

class PITGEncoder(nn.Module):
    def __init__(self, e1_vocab_size, e2_vocab_size, e3_vocab_size, e1_hidden=128, e2_hidden=16, e3_hidden=16, num_layers = 3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(PITGEncoder, self).__init__()
        self.num_layers = num_layers
        # Password encoder
        self.e1 = LSTMEncoder(input_size = self.e1_vocab_size, hidden_size = e1_hidden, num_layers=num_layers)
        # Region encoder
        self.e2 = EntityEncoder(input_size = self.e2_vocab_size, hidden_size=e2_hidden, num_layers=num_layers)
        # Host encoder
        self.e3 = EntityEncoder(input_size = self.e3_vocab_size, hidden_size=e3_hidden, num_layers=num_layers)

    def forward(self, x_input, region_input, host_input, encoder_hidden=None):
        h1 = self.e1(x_input, encoder_hidden)
        h2 = self.e2(region_input)
        h3 = self.e3(host_input)
        return (torch.cat([h1[0], h2[0], h3[0]], dim=1), torch.cat([h1[1], h2[1], h3[1]], dim=1))

class PITGModel(nn.Module):
    def __init__(self, e1_vocab_size, e2_vocab_size, e3_vocab_size, trans_size, pad_token,start_token,end_token,e1_hidden=128, e2_hidden=16, e3_hidden=16, 
                 embed_size=200, maxlen=32, num_layers=1, device=torch.device('cpu')):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(PITGModel, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.e1_vocab_size = e1_vocab_size
        self.e2_vocab_size = e2_vocab_size
        self.e3_vocab_size = e3_vocab_size

        self.output_size = trans_size
        self.e1_hidden = e1_hidden
        self.e2_hidden = e2_hidden
        self.e3_hidden = e3_hidden
        self.embed_size = embed_size
        self.mode = "train"
        self.maxlen = maxlen
        self.device = device
        self.num_layers = num_layers
    
        self.encoder = PITGEncoder(e1_vocab_size, e2_vocab_size, e3_vocab_size,e1_hidden=128, e2_hidden=16, e3_hidden=16,num_layers=num_layers)
        self.decoder = LSTMDecoder(input_size = self.input_size, decode_size=self.output_size, hidden_size = (e1_hidden+e2_hidden+e3_hidden), num_layers=num_layers)

    def set_mode(self, mode):
        self.mode = mode

    def forward_train(self, pwds, region_input, host_input, edits):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_hidden = self.encoder(pwds, region_input, host_input, encoder_hidden)
        decoder_input = self.decoder.init_start_token(batch_size, self.start_token, self.device)
        decoder_hidden = encoder_hidden
        maxlen = edits.size(1)
        outputs = torch.zeros(batch_size, maxlen, self.vocab_size).to(self.device)
        for t in range(maxlen): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            decoder_input = edits[:, t].unsqueeze(1)
        return outputs

    def forward_greedy_predict(self, pwds, region_input, host_input):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_hidden = self.encoder(pwds, region_input, host_input, encoder_hidden)
        decoder_input = self.decoder.init_start_token(batch_size, self.start_token, self.device)
        decoder_hidden = encoder_hidden
        maxlen = self.maxlen
        outputs = [[] for _ in range(batch_size)]
        mask = torch.ones(batch_size)
        for t in range(maxlen): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            top_probs, top_ids = decoder_output.squeeze(1).topk(1, dim=1)
            decoder_input = top_ids
            top_probs = top_probs.squeeze(1)
            top_ids = top_ids.squeeze(1)
            for i in range(batch_size):
                if mask[i] == 0:
                    continue
                if top_ids[i] == self.end_token:
                    mask[i] = 0
                    continue
                outputs[i].append((top_ids[i].item(), -top_probs[i].item()))
            if torch.sum(mask) == 0:
                break
        return outputs
    
    def forward_beamsearch(self, pwds, region_input, host_input, beamwidth=10, topk=10, max_predict=8):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_hidden = self.encoder(pwds, region_input, host_input, encoder_hidden)
        encoder_hc = encoder_hidden[0]
        encoder_hx = encoder_hidden[1]
        maxlen = self.maxlen
        outputs = [None for _ in range(batch_size)]
        for t in range(batch_size):
            decoder_hc = encoder_hc[:, t:t+1]
            decoder_hx = encoder_hx[:, t:t+1]
            root = (decoder_hc, decoder_hx, None, self.start_token, 0)

            queue = [root]
            end_nodes = []
            predict_cnt = 0
            while max_predict > predict_cnt and len(queue) > 0 and len(end_nodes) < topk:
                decoder_hc = torch.cat([node[0] for node in queue], dim=1)
                decoder_hx = torch.cat([node[1] for node in queue], dim=1)
                decoder_hidden = (decoder_hc, decoder_hx)
                decoder_input = torch.tensor([node[3] for node in queue]).unsqueeze(1).to(self.device)
                decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hc, decoder_hx))
                decoder_hc = decoder_hidden[0]
                decoder_hx = decoder_hidden[1]
                topv, topi = decoder_output.squeeze(1).topk(beamwidth, dim=1)
                topv = -topv
                topv = topv.tolist()
                topi = topi.tolist()
                mid_nodes = []
                for i in range(len(queue)):
                    node = queue[i]
                    hc = decoder_hc[:, i:i+1]
                    hx = decoder_hx[:, i:i+1]
                    for j in range(beamwidth):
                        topi_v = topi[i][j]
                        logp = node[-1]+topv[i][j]
                        cur = (hc, hx, node, topi_v, logp)
                        if topi[i][j] == self.end_token:
                            end_nodes.append(cur)
                            if len(end_nodes) >= topk:
                                break
                        else:
                            mid_nodes.append(cur)
                    if len(end_nodes) >= topk:
                        break
                if len(mid_nodes) > beamwidth:
                    queue = heapq.nsmallest(beamwidth, mid_nodes, key=lambda x:x[-1])
                else:
                    queue = mid_nodes
                predict_cnt += 1
            guesses = []
            for node in end_nodes:
                edits = []
                cur = node 
                while cur[2] != None:
                    edits.append((cur[3]))
                    cur = cur[2]
                edits = edits[::-1]
                guesses.append((edits, node[-1]))
            outputs[t] = sorted(guesses, key=lambda x:x[1])
        return outputs

    def forward(self, pwds, x_length, edits=None, beamwidth=10, topk=10, max_queue_size=20000):
        ans = None
        
        if self.mode == "train":
            ans = self.forward_train(pwds, x_length, edits)
        elif self.mode == "predict":
            with torch.no_grad():
                ans = self.forward_greedy_predict(pwds, x_length)
        elif self.mode == "beamsearch":
            with torch.no_grad():
                ans = self.forward_beamsearch(pwds, x_length, beamwidth=beamwidth, topk=topk, max_queue_size=max_queue_size)
        return ans
