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
from queue import PriorityQueue

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 定义编码器类
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        _, hidden = self.lstm(embedded)
        return hidden

# 定义解码器类
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        output = self.softmax(output)
        return output, hidden

# 定义 Seq2Seq 模型类
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, teacher_forcing_ratio=1.0):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.teacher_forcing_ratio=teacher_forcing_ratio

    def forward(self, input_seq, target_seq):
        batch_size = input_seq.size(0)
        target_len = target_seq.size(1)
        target_vocab_size = self.decoder.fc.out_features

        encoder_hidden = self.encoder(input_seq)

        decoder_input = torch.tensor([[1]] * batch_size)  # 设置起始符号的索引为 0
        decoder_hidden = encoder_hidden

        outputs = torch.zeros(batch_size, target_len, target_vocab_size)

        for t in range(0, target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            use_teacher_forcing = True if torch.rand(1) < self.teacher_forcing_ratio else False
            if use_teacher_forcing:
                decoder_input = target_seq[:, t].unsqueeze(1)
            else:
                decoder_input = decoder_output.argmax(dim=2)

        return outputs

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb

    def eval(self):
        return self.logp
    
    def __lt__(self, node):
        return self.logp < node.logp

class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 3, embed_size=200, dropout=0.3):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings = input_size, embedding_dim = self.hidden_size)
        self.lstm = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, num_layers = num_layers, batch_first=True)
    
    def forward(self, x_input, x_length=None, encoder_hidden=None):
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
        # return (0.1*torch.ones(self.num_layers, batch_size, self.hidden_size).to(device),
        #         0.1*torch.ones(self.num_layers, batch_size, self.hidden_size).to(device))
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        # return (torch.randn(self.num_layers, batch_size, self.hidden_size).to(device),
        #         torch.randn(self.num_layers, batch_size, self.hidden_size).to(device))

class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, input_size, decode_size, hidden_size, num_layers = 3, embed_size=128, dropout=0.3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(lstm_decoder, self).__init__()
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


def get_tensors(input_tuple, index):
    tensor1, tensor2 = input_tuple
    vector1 = tensor1[:, index, :].unsqueeze(1).contiguous()  # 在第0维添加一个维度
    vector2 = tensor2[:, index, :].unsqueeze(1).contiguous()  # 在第0维添加一个维度
    return (vector1, vector2)

class Pass2PathModel(nn.Module):
    def __init__(self, vocab_size, trans_size, pad_token,start_token,end_token,hidden_size=128, embed_size=200, maxlen=32, num_layers=1,device=torch.device('cpu')):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(Pass2PathModel, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.input_size = vocab_size
        self.output_size = trans_size
        self.hidden_size = hidden_size
        self.vocab_size = trans_size
        self.embed_size = embed_size
        self.mode = "train"
        self.maxlen = maxlen
        self.device = device
    
        self.encoder = lstm_encoder(input_size = self.input_size, hidden_size = hidden_size, num_layers=num_layers)
        self.decoder = lstm_decoder(input_size = self.input_size, decode_size=self.vocab_size, hidden_size = hidden_size, num_layers=num_layers)

    def set_mode(self, mode):
        self.mode = mode

    def forward_train(self, pwds, x_length, edits):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_hidden = self.encoder(pwds, x_length)
        decoder_input = self.decoder.init_start_token(batch_size, self.start_token, self.device)
        decoder_hidden = encoder_hidden
        maxlen = edits.size(1)
        outputs = torch.zeros(batch_size, maxlen, self.vocab_size).to(self.device)
        for t in range(maxlen): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            decoder_input = edits[:, t].unsqueeze(1)
        return outputs

    def forward_greedy_predict(self, pwds, x_length):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_output, encoder_hidden = self.encoder(pwds, x_length, encoder_hidden)
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
                outputs[i].append((top_ids[i].item(), top_probs[i].item()))
            if torch.sum(mask) == 0:
                break
        return outputs
    
    def forward_beamsearch(self, pwds, x_length, beamwidth=10, topk=10, max_queue_size=20000):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)
        encoder_output, encoder_hidden = self.encoder(pwds, x_length, encoder_hidden)
        maxlen = self.maxlen
        outputs = [None for _ in range(batch_size)]
        for t in range(batch_size):
            decoder_input = self.decoder.init_start_token(1, self.start_token, self.device)
            decoder_hidden = get_tensors(encoder_hidden, t)
            # decoder_hidden = self.encoder.init_hidden(1, self.device)
            print(">>> ", torch.sum(decoder_hidden[0]))
            end_nodes = []
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0)
            nodes = PriorityQueue()
            nodes.put((node.logp, node))
            while nodes.qsize() <= max_queue_size and nodes.qsize() > 0:
                score, node = nodes.get()
                decoder_input = node.wordid
                decoder_hidden = node.h
                if node.wordid.squeeze().item() == self.end_token:
                    end_nodes.append(node)
                    if len(end_nodes) >= topk:
                        break
                    continue 
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, top_ids = decoder_output.squeeze(1).topk(beamwidth, dim=1)
                logps = -torch.log(topv)
                for i in range(beamwidth):
                    decode_id = top_ids[:, i].unsqueeze(1)
                    logp = logps[0, i].item() + score
                    next_node = BeamSearchNode(decoder_hidden, node, decode_id, logp)
                    nodes.put((logp, next_node))
            guesses = []
            for node in sorted(end_nodes, key=lambda x:x.logp):
                edits = []
                cur = node 
                while cur.prevNode != None:
                    edits.append((cur.wordid.squeeze().item()))
                    cur = cur.prevNode
                edits = edits[::-1]
                guesses.append((edits, node.logp))
            outputs[t] = sorted(guesses, key=lambda x:x[1])
        return outputs

    def forward(self, pwds, x_length, edits=None, beamwidth=10, topk=10, max_queue_size=20000):
        if self.mode == "train":
            return self.forward_train(pwds, x_length, edits)
        elif self.mode == "predict":
            return self.forward_greedy_predict(pwds, x_length)
        elif self.mode == "beamsearch":
            return self.forward_beamsearch(pwds, x_length, beamwidth=beamwidth, topk=topk, max_queue_size=max_queue_size)


