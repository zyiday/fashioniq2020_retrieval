# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for text data."""
import string
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

import torch_functions
import pdb


class SimpleVocab(object):

  def __init__(self):
    super(SimpleVocab, self).__init__()
    self.word2id = {}
    self.wordcount = {}
    self.word2id['<UNK>'] = 0
    self.word2id['<AND>'] = 1
    self.word2id['<BOS>'] = 2
    self.word2id['<EOS>'] = 3
    self.wordcount['<UNK>'] = 9e9
    self.wordcount['<AND>'] = 9e9
    self.wordcount['<BOS>'] = 9e9
    self.wordcount['<EOS>'] = 9e9

  def tokenize_text(self, text):
    tokens = text.split()
    return tokens

  def add_text_to_vocab(self, text):
    tokens = self.tokenize_text(text)
    for token in tokens:
      if not token in self.word2id:
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 0
      self.wordcount[token] += 1

  def threshold_rare_words(self, wordcount_threshold=2):
    for w in self.word2id:
      if self.wordcount[w] < wordcount_threshold:
        self.word2id[w] = 0

  def encode_text(self, text):
    tokens = self.tokenize_text(text)
    x = [self.word2id.get(t, 0) for t in tokens]
    return x

  def get_size(self):
    return len(self.word2id)

  def get_real_size(self):
    count = 0
    for w in self.word2id:
      if self.word2id[w] != 0:
        count += 1
    return count


class TextLSTMModel(torch.nn.Module):

  def __init__(self,
               texts_to_build_vocab=None,
               word_embed_dim=512,
               lstm_hidden_dim=512):

    super(TextLSTMModel, self).__init__()

    self.vocab = SimpleVocab()
    if texts_to_build_vocab != None:
      for text in texts_to_build_vocab:
        self.vocab.add_text_to_vocab(text)
      # out = {}
      # out['word2id'] = self.vocab.word2id
      # out['wordcount'] = self.vocab.wordcount
      # json.dump(out, open("simplevocab.json", "w"))
    else:
      vocab_data = json.load(open("simplevocab.json"))
      self.vocab.word2id = vocab_data['word2id']
      self.vocab.wordcount = vocab_data['wordcount']
    
    self.vocab.threshold_rare_words(wordcount_threshold=3)
    vocab_size = self.vocab.get_size()
    self.word_embed_dim = word_embed_dim
    self.lstm_hidden_dim = lstm_hidden_dim
    self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
    self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
    self.fc_output = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
    )
    # self.init_glove()

  def init_glove(self):
    self.embedding_layer.weight.data = torch.load("glove_840b.th")

  def forward(self, x, return_mp=False):
    """ input x: list of strings"""
    if type(x) is list:
      if type(x[0]) is str or type(x[0]) is unicode:
        x = [self.vocab.encode_text(text) for text in x]

    assert type(x) is list
    assert type(x[0]) is list
    assert type(x[0][0]) is int
    return self.forward_encoded_texts(x, return_mp)

  def forward_encoded_texts(self, texts, return_mp):
    # to tensor
    lengths = [len(t) for t in texts]
    itexts = torch.zeros((np.max(lengths), len(texts))).long()
    for i in range(len(texts)):
      itexts[:lengths[i], i] = torch.tensor(texts[i])

    # embed words
    itexts = torch.autograd.Variable(itexts).cuda()
    etexts = self.embedding_layer(itexts)

    # lstm
    lstm_output, _ = self.forward_lstm_(etexts)
    lstm_output = self.fc_output(lstm_output)

    # get last output (using length)
    text_features = []
    for i in range(len(texts)):
      text_features.append(lstm_output[lengths[i] - 1, i, :])

    # output
    text_features = torch.stack(text_features)

    if return_mp:
      return lstm_output, lengths, text_features
    return text_features

  def forward_lstm_(self, etexts):
    batch_size = etexts.shape[1]
    first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                    torch.zeros(1, batch_size, self.lstm_hidden_dim))
    first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
    lstm_output, last_hidden = self.lstm(etexts, first_hidden)
    return lstm_output, last_hidden

class TextDualencModel(torch.nn.Module):

  def __init__(self,
               texts_to_build_vocab,
               embed_dim=512, 
               word_embed_dim=300,
               lstm_hidden_dim=512):

    super(TextDualencModel, self).__init__()

    self.vocab = SimpleVocab()
    if texts_to_build_vocab != None:
      for text in texts_to_build_vocab:
        self.vocab.add_text_to_vocab(text)
      # out = {}
      # out['word2id'] = self.vocab.word2id
      # out['wordcount'] = self.vocab.wordcount
      # json.dump(out, open("simplevocab.json", "w"))
    else:
      vocab_data = json.load(open("simplevocab.json"))
      self.vocab.word2id = vocab_data['word2id']
      self.vocab.wordcount = vocab_data['wordcount']
    self.vocab.threshold_rare_words(wordcount_threshold=3)
  
    self.rnn_type = 'gru'
    self.rnn_size = lstm_hidden_dim
  
    self.vocab_size = self.vocab.get_size()
    self.word_dim = word_embed_dim
    self.embed_size = embed_dim
    self.num_layers = 1
    self.bidirectional = True
    self.ft_norm = True
    self.batchnorm = True
    self.tanh = True
    self.dropout = 0.2

    # 1, 2, 3 MP, bi-GRU, biGRU-CNN
    self.level1 = 1 # 0, 1
    self.level2 = 1
    self.level3 = 1
    self.r = 512
    self.max_words_in_sent = 30

    self.embed = nn.Embedding(self.vocab_size, self.word_dim)
    
    self.fc_size = self.level1 * self.word_dim + self.level2 * self.rnn_size * 2 + self.level3 * self.r * 3
    
    if self.level2 or self.level3:
      self.rnn = torch_functions.rnn_factory(self.rnn_type,
        input_size=self.word_dim, hidden_size=self.rnn_size, 
        num_layers=self.num_layers, bidirectional=self.bidirectional, 
        bias=True, batch_first=True)

    if self.level3:
      self.conv2 = nn.Conv1d(self.rnn_size * 2, self.r, 2)
      self.conv3 = nn.Conv1d(self.rnn_size * 2, self.r, 3)
      self.conv4 = nn.Conv1d(self.rnn_size * 2, self.r, 4)

    self.fc = nn.Linear(self.fc_size, self.embed_size)

    if self.batchnorm:
      self.bn = nn.BatchNorm1d(self.embed_size)

    self.dropout = nn.Dropout(p=self.dropout)
    self.init_weights()

  def xavier_init_fc(self, fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

  def init_weights(self):
    """Xavier initialization for the fully connected layer
    """
    self.embed.weight.data = torch.load("glove_840b.th")
    # self.embed.weight.data.uniform_(-0.1, 0.1)
    self.xavier_init_fc(self.fc)

  def forward(self, x):
    if type(x) is list:
      if type(x[0]) is str or type(x[0]) is unicode:
        texts = [self.vocab.encode_text(t) for t in x]
    assert type(texts) is list
    assert type(texts[0]) is list
    assert type(texts[0][0]) is int

    # to tensor
    lengths = [len(t) for t in texts]
    itexts = torch.zeros((len(texts), np.max(lengths))).long()
    for i in range(len(texts)):
      itexts[i, :lengths[i]] = torch.tensor(texts[i])
    # embed words
    itexts = torch.autograd.Variable(itexts).cuda()
    lengths = torch.LongTensor(lengths).cuda()

    # Embed word ids to vectors
    inputs = self.embed(itexts)
    features = []
    seq_masks = torch_functions.sequence_mask(lengths, max_len=inputs.size(1)).float()
    
    if self.level1:
      feature = torch.sum(inputs * seq_masks.unsqueeze(2), 1) / torch.sum(seq_masks, 1, keepdim=True)
      features.append(feature)

    if self.level2 or self.level3:
      inputs = self.dropout(inputs)
      outs, states = torch_functions.calc_rnn_outs_with_sort(self.rnn, inputs, lengths)

    if self.level2:
      feature = torch.sum(outs * seq_masks.unsqueeze(2), 1) / torch.sum(seq_masks, 1, keepdim=True)
      feature = self.dropout(feature)
      features.append(feature)

    if self.level3:
      outs = outs * seq_masks.unsqueeze(2)
      hc_conv = outs.permute(0, 2, 1)
      c2 = self.conv2(hc_conv)
      c3 = self.conv3(hc_conv)
      c4 = self.conv4(hc_conv)
      c2 = F.relu(c2)
      c3 = F.relu(c3)
      c4 = F.relu(c4)
      c2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
      c3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
      c4 = F.max_pool1d(c4, c4.size(2)).squeeze(2)
      feature = torch.cat([c2, c3, c4], 1)
      feature = self.dropout(feature)
      features.append(feature)

      ft = torch.cat(features, 1)

      ft = self.fc(ft)

      if self.batchnorm:
        ft = self.bn(ft)

      if self.tanh:
        ft = torch.tanh(ft)

      ft = self.dropout(ft)

      if self.ft_norm:
        ft = torch_functions.l2norm(ft)

    return ft