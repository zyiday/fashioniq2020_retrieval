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

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm

import pdb

dataname2id = {'dress': 0, 'shirt': 1, 'toptee': 2}
datanames = ['dress', 'shirt', 'toptee']

def test(opt, model, testset, dataname):
  """Tests a model over the given testset."""
  model.eval()
  dataname_id = dataname2id[dataname]
  test_queries = testset.get_test_queries()[dataname_id]
  test_targets = testset.get_test_targets()[dataname_id]

  all_queries = []
  all_imgs = []
  if test_queries:
    # compute test query features
    imgs = []
    mods = []
    for t in tqdm(test_queries):
      imgs += [t['source_img_data']]
      mods += [t['mod']['str']]
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float().cuda()
        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
    all_queries = np.concatenate(all_queries)
    
    # compute all image features
    imgs = []
    logits = []
    for t in tqdm(test_targets):
      imgs += [t['target_img_data']]
      if len(imgs) >= opt.batch_size or t is test_targets[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float().cuda()
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    
  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)

  for i, t in enumerate(test_queries):
    sims[i, t['source_img_id']] = -10e10  # remove query image

  if opt.return_test_rank:
    return sims

  nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

  # compute recalls
  out = []
  for k in [10, 50]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if test_queries[i]['target_img_id'] in nns[:k]:
        r += 1
    r = 100 * r / len(nn_result)
    out += [('{}_r{}'.format(dataname, k), r)]
  return out, sims
