
# TODO(lujiang): put it into the third-party
# MIT License

# Copyright (c) 2018 Nam Vo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Metric learning functions.

Codes are modified from:
https://github.com/lugiavn/generalization-dml/blob/master/nams.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def pairwise_distances(x, y=None):
  """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  return torch.clamp(dist, 0.0, np.inf)


class MyTripletLossFunc(torch.autograd.Function):

  def __init__(self, triplets):
    super(MyTripletLossFunc, self).__init__()
    self.triplets = triplets
    self.triplet_count = len(triplets)

  def forward(self, features):
    self.save_for_backward(features)

    self.distances = pairwise_distances(features).cpu().numpy()

    loss = 0.0
    triplet_count = 0.0
    correct_count = 0.0
    for i, j, k in self.triplets:
      w = 1.0
      triplet_count += w
      loss += w * np.log(1 +
                         np.exp(self.distances[i, j] - self.distances[i, k]))
      if self.distances[i, j] < self.distances[i, k]:
        correct_count += 1

    loss /= triplet_count
    return torch.FloatTensor((loss,))

  def backward(self, grad_output):
    features, = self.saved_tensors
    features_np = features.cpu().numpy()
    grad_features = features.clone() * 0.0
    grad_features_np = grad_features.cpu().numpy()

    for i, j, k in self.triplets:
      w = 1.0
      f = 1.0 - 1.0 / (
          1.0 + np.exp(self.distances[i, j] - self.distances[i, k]))
      grad_features_np[i, :] += w * f * (
          features_np[i, :] - features_np[j, :]) / self.triplet_count
      grad_features_np[j, :] += w * f * (
          features_np[j, :] - features_np[i, :]) / self.triplet_count
      grad_features_np[i, :] += -w * f * (
          features_np[i, :] - features_np[k, :]) / self.triplet_count
      grad_features_np[k, :] += -w * f * (
          features_np[k, :] - features_np[i, :]) / self.triplet_count

    for i in range(features_np.shape[0]):
      grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
    grad_features *= float(grad_output.data[0])
    return grad_features


class TripletLoss(torch.nn.Module):
  """Class for the triplet loss."""
  def __init__(self, pre_layer=None):
    super(TripletLoss, self).__init__()
    self.pre_layer = pre_layer

  def forward(self, x, triplets):
    if self.pre_layer is not None:
      x = self.pre_layer(x)
    loss = MyTripletLossFunc(triplets)(x)
    return loss


class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features


def l2norm(inputs, dim=-1):
  # inputs: (batch, dim_ft)
  norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
  inputs = inputs / norm
  return inputs

def sequence_mask(lengths, max_len=None):
  ''' Creates a boolean mask from sequence lengths.
  '''
  # lengths: LongTensor, (batch, )
  batch_size = lengths.size(0)
  max_len = max_len or lengths.max()
  return (torch.arange(0, max_len)
          .type_as(lengths)
          .repeat(batch_size, 1)
          .lt(lengths.unsqueeze(1)))

        
def rnn_factory(rnn_type, **kwargs):
  # Use pytorch version when available.
  rnn = getattr(nn, rnn_type.upper())(**kwargs)
  return rnn


def calc_rnn_outs_with_sort(rnn, inputs, seq_lens, init_states=None):
  '''
  inputs: FloatTensor, (batch, seq_len, dim_ft)
  seq_lens: LongTensor, (batch,)
  '''
  seq_len = inputs.size(1)
  # sort
  sorted_seq_lens, seq_sort_idx = torch.sort(seq_lens, descending=True)
  _, seq_unsort_idx = torch.sort(seq_sort_idx, descending=False)
  # pack
  inputs = torch.index_select(inputs, 0, seq_sort_idx)
  packed_inputs = pack_padded_sequence(inputs, sorted_seq_lens, batch_first=True)
  if init_states is not None:
    if isinstance(init_states, tuple):
      new_states = []
      for i, state in enumerate(init_states):
        new_states.append(torch.index_select(state, 1, seq_sort_idx))
      init_states = tuple(new_states)
    else:
      init_states = torch.index_select(init_states, 1, seq_sort_idx)
  # rnn
  packed_outs, states = rnn(packed_inputs, init_states)
  # unpack
  outs, _ = pad_packed_sequence(packed_outs, batch_first=True, 
    total_length=seq_len, padding_value=0)
  # unsort
  # outs.size = (batch, seq_len, num_directions * hidden_size)     
  outs = torch.index_select(outs, 0, seq_unsort_idx)   
  if isinstance(states, tuple):
    # states: (num_layers * num_directions, batch, hidden_size)
    new_states = []
    for i, state in enumerate(states):
      new_states.append(torch.index_select(state, 1, seq_unsort_idx))
    states = tuple(new_states)
  else:
    states = torch.index_select(states, 1, seq_unsort_idx)

  return outs, states