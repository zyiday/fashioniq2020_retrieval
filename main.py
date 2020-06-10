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

"""Main method to train the model."""


#!/usr/bin/python

import argparse
import sys
import time
import datasets
import img_text_composition_models
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
from torch.utils.data import dataloader
import torchvision
from tqdm import tqdm as tqdm
import pdb

torch.set_num_threads(3)

def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='fashioniq')
  parser.add_argument('--root_dir', type=str, default='./data/{}')
  parser.add_argument('--log_dir', type=str, default='./runs')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--img_encoder', type=str, default='efficientnet')
  parser.add_argument('--text_encoder', type=str, default='dualenc')
  parser.add_argument('--embed_dim', type=int, default=1024)
  parser.add_argument('--optimizer', type=str, default='Adam') # SGD Adam
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument(
      '--learning_rate_decay_patient', type=int, default=5)
  parser.add_argument('--eval_frequency', type=int, default=1)
  parser.add_argument('--lr_div', type=float, default=0.5) # 0.1
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_epoch', type=int, default=30)
  parser.add_argument('--loss', type=str, default='batch_based_classification')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  parser.add_argument('--is_test', default=False, action='store_true')
  parser.add_argument('--return_test_rank', default=False, action='store_true')
  parser.add_argument('--resume_file', default=None)
  args = parser.parse_args()
  return args

def load_dataset(opt):
  """Loads the input datasets."""
  print('Reading dataset ', opt.dataset)
  if opt.dataset == 'fashioniq':
    trainset = datasets.FashionIQ(
        anno_dir=opt.root_dir.format('annotation'),
        image_dir=opt.root_dir.format('images'),
        split_dir=opt.root_dir.format('public_split'),
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),            
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
        
    testset = datasets.FashionIQ(
        anno_dir=opt.root_dir.format('annotation'),
        image_dir=opt.root_dir.format('images'),
        split_dir=opt.root_dir.format('public_split'),
        split='val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  else:
    print('Invalid dataset', opt.dataset)
    sys.exit()

  print('trainset size:', len(trainset))
  print('testset size:', len(testset))
  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print('Creating model and optimizer for', opt.model)
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, opt)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, opt)
  elif opt.model == 'add':
    model = img_text_composition_models.Add(texts, opt)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, opt)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(texts, opt)
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, opt)
  else:
    print('Invalid model', opt.model)
    print('available: imgonly, textonly, add, concat, tirg, tirg_lastconv')
    sys.exit()
  model = model.cuda()
  
  # create optimizer
  params = []
  per_params = []
  for name, param in model.named_parameters():
    per_params.append(param)
  params.append({'params': per_params})
  if opt.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
  elif opt.optimizer == 'Adam':
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
  return model, optimizer


def train_loop(opt, logger, trainset, testset, model, optimizer, DP_data=None):
  """Function for train loop"""
  print('Begin training')
  losses_tracking = {}
  best_eval = 0
  it = 0
  tic = time.time()
  
  for epoch in range(opt.num_epoch):
    # decay learing rate epoch
    if epoch != 0 and epoch % opt.learning_rate_decay_patient == 0:
      for g in optimizer.param_groups:
        g['lr'] *= opt.lr_div

    # run trainning for 1 epoch
    model.train()
    trainloader = dataloader.DataLoader(trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)

    def training_1_iter(data, data_dp=None):
      img1 = data['source_img_data'].cuda()
      img2 = data['target_img_data'].cuda()
      mods = data['mod']['str']

      # compute loss
      losses = []
      if opt.loss == 'batch_based_classification':
        loss_value = model.compute_loss(
            img1, mods, img2, soft_triplet_loss=False)
      else:
        print('Invalid loss function', opt.loss)
        sys.exit()
      loss_name = opt.loss
      loss_weight = 1.0
      losses += [(loss_name, loss_weight, loss_value)]

      total_loss = sum([
          loss_weight * loss_value
          for loss_name, loss_weight, loss_value in losses
      ])
      assert not torch.isnan(total_loss)
      losses += [('total training loss', None, total_loss)]

      # track losses
      for loss_name, loss_weight, loss_value in losses:
        if not loss_name in losses_tracking:
          losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value))

      # gradient descend
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
    
    count_dp_idx = 0
    for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
      it += 1
      training_1_iter(data)
    
    # show/log stats
    print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic, 4))
    tic = time.time()
    for loss_name in losses_tracking:
      avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
      print('    Loss', loss_name, round(avg_loss, 4))
      logger.add_scalar(loss_name, avg_loss, it)
    logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

    # test
    if epoch % opt.eval_frequency == 0:
      tests = []
      all_sims = {}
      rsum = 0
      for dataname in testset.data_name:
        t, sims = test_retrieval.test(opt, model, testset, dataname)
        all_sims[dataname] = sims
        for metric_name, metric_value in t:
          tests += [(metric_name, metric_value)]
          rsum += metric_value
      tests += [('rmean', rsum / 6)]
      for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print('    ', metric_name, round(metric_value, 2))

      if rsum > best_eval:
        best_eval = rsum
        # save checkpoint
        for dataname in testset.data_name:
          np.save(opt.log_dir + '/val.{}.{}.scores.npy'.format(dataname, opt.model), all_sims[dataname])
        torch.save({
            'it': it,
            'opt': opt,
            'model_state_dict': model.state_dict(),
        },
        logger.file_writer.get_logdir() + '/best_checkpoint.pth')

  print('Finished training')

def test(opt, testset, model):
  print('Begin testing')
  # test for submit
  if opt.return_test_rank:
    for dataname in testset.data_name:
      sims = test_retrieval.test(opt, model, testset, dataname)
      np.save(opt.log_dir + '/test.{}.{}.scores.npy'.format(dataname, opt.model), sims)
    exit()

  tests = []
  rsum = 0
  for dataname in testset.data_name:
    t, sims = test_retrieval.test(opt, model, testset, dataname)
    np.save(opt.log_dir + '/val.{}.{}.scores.npy'.format(dataname, opt.model), sims)
    for metric_name, metric_value in t:
      tests += [(metric_name, metric_value)]
      rsum += metric_value
  tests += [('rmean', rsum / 6)]

  for metric_name, metric_value in tests:
    print('    ', metric_name, round(metric_value, 2))
  print('Finished testing')

def load_state_dicts(opt, model):
  state_dict = torch.load(opt.resume_file, map_location=lambda storage, loc: storage)['model_state_dict']
  num_resumed_vars = 0
  own_state_dict = model.state_dict()
  new_state_dict = {}
  for varname, varvalue in state_dict.items():
    if varname in own_state_dict:
      new_state_dict[varname] = varvalue
      num_resumed_vars += 1
  own_state_dict.update(new_state_dict)
  model.load_state_dict(own_state_dict)
  print('number of resumed variables: {}'.format(num_resumed_vars))

def main():
  opt = parse_opt()
  print('Arguments:')
  for k in opt.__dict__.keys():
    print('    ', k, ':', str(opt.__dict__[k]))
    
  if not opt.is_test:
    logger = SummaryWriter(opt.log_dir)
    print('Log files saved to', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
      logger.add_text(k, str(opt.__dict__[k]))

    trainset, testset = load_dataset(opt)
    model, optimizer = create_model_and_optimizer(opt, None)
    if opt.resume_file != None:
      load_state_dicts(opt, model)
    
    train_loop(opt, logger, trainset, testset, model, optimizer)    
    logger.close()

  else:
    tmp_split = 'val'
    if opt.return_test_rank:
      tmp_split = 'test'
    
    testset = datasets.FashionIQ(
      anno_dir=opt.root_dir.format('annotation'),
      image_dir=opt.root_dir.format('images'),
      split_dir=opt.root_dir.format('public_split'),
      split=tmp_split,
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize(256),
          torchvision.transforms.CenterCrop(224),
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
      ]))
    model, optimizer = create_model_and_optimizer(opt, None)
    state_dicts = torch.load(opt.resume_file, map_location=lambda storage, loc: storage)['model_state_dict']
    model.load_state_dict(state_dicts)
    test(opt, testset, model)


if __name__ == '__main__':
  main()
