# Copyright 2019 Google Inc. All Rights Reserved.
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

"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import nltk
import pdb


class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset."""

  def __init__(self):
    super(BaseDataset, self).__init__()
    self.test_targets = []
    self.test_queries = []

  def get_loader(self,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0):
    return torch.utils.data.DataLoader(
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda i: i)

  def get_test_queries(self):
    return self.test_queries
  
  def get_test_targets(self):
    return self.test_targets

  def get_all_texts(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    return self.generate_random_query_target()

  def generate_random_query_target(self):
    raise NotImplementedError

  def get_img(self, idx, raw_img=False):
    raise NotImplementedError


class FashionIQ(BaseDataset):
  """FashionIQ dataset."""
  def __init__(self, anno_dir, image_dir, split_dir, split='train', transform=None):
    super(FashionIQ, self).__init__()
    self.anno_dir = anno_dir
    self.image_dir = image_dir
    self.split_dir = split_dir
    self.split = split
    self.transform = transform
    self.data_name = ['dress', 'shirt', 'toptee']

    self.ref_captions, self.caps_splitpoint = [], []
    for idx, name in enumerate(self.data_name):
      self.caps_splitpoint.append(len(self.ref_captions))
      tmp_captions = json.load(open("{}/ref_caption_{}_{}.json".format(anno_dir, name, split)))
      self.ref_captions.extend(tmp_captions)
    self.num_caption = len(self.ref_captions)
    self.caps_splitpoint.append(self.num_caption)
    
    # names
    self.names, self.num_image = [], []
    for name in self.data_name:
      self.names.append(json.load(open("{}/split.{}.{}.json".format(split_dir, name, split))))
      self.num_image.append(len(self.names[-1]))
    
    print('captions size %d' % (self.num_caption))

    if split != 'train':
      self.generate_test_queries_()
      self.generate_test_targets_()

  def concat_text(self, captions):
    text = "<BOS> {} <AND> {} <EOS>".format(captions[0], captions[1])
    return text

  def generate_test_queries_(self):
    all_captions = []
    for idx, name in enumerate(self.data_name):
      all_captions.append(json.load(open("{}/ref_caption_{}_{}.json".format(self.anno_dir, name, self.split))))

    for i in range(len(all_captions)):
      tmp = []
      for idx in range(len(all_captions[i])):
        caption = all_captions[i][idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        if self.split != 'test':
          target = caption['target']

        out = {}
        out['source_img_id'] = self.names[i].index(candidate)
        out['source_img_data'] = self.get_img(candidate)
        if self.split != 'test':
          out['target_img_id'] = self.names[i].index(target)
          out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        
        tmp += [out]
      self.test_queries.append(tmp)

  def generate_test_targets_(self):
    for name in self.names:
      tmp = []
      for idx in range(len(name)):
        target = name[idx]
        out = {}
        out['target_img_id'] = idx
        out['target_img_data'] = self.get_img(target)
        tmp += [out]
      self.test_targets.append(tmp)

  def get_all_texts(self):
    all_captions = []
    for idx, name in enumerate(self.data_name):
      all_captions.extend(json.load(open("{}/ref_caption_{}_{}.json".format(self.anno_dir, name, 'train'))))
    texts = []
    for caption in all_captions:
      sent_list = caption['captions']
      for c in sent_list:
        texts.append(c)
    return texts

  def __len__(self):
      return len(self.ref_captions)

  def __getitem__(self, idx):     
    caption = self.ref_captions[idx]
    mod_str = self.concat_text(caption['captions'])
    candidate = caption['candidate']
    if self.split != 'test':
      target = caption['target']

    out = {}
    out['source_img_data'] = self.get_img(candidate)
    if self.split != 'test':
      out['target_img_data'] = self.get_img(target)
    out['mod'] = {'str': mod_str}

    return out

  def get_img(self, img_name, raw_img=False):
    img_path = os.path.join(self.image_dir, img_name + ".jpg")
    with open(img_path, 'rb') as f:
      img = PIL.Image.open(f)
      img = img.convert('RGB')
    if raw_img:
      return img
    if self.transform:
      img = self.transform(img)
    return img
