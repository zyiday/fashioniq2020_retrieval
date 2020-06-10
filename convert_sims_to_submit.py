import os
import numpy as np
import json
import argparse
import pdb

dataname2id = {'dress': 0, 'shirt': 1, 'toptee': 2}
datanames = ['dress', 'shirt', 'toptee']


def convert_sims_to_submit(sims, dataname, captions, test_names):
  nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
  out = captions[dataname2id[dataname]]
  idx2name = np.array(test_names[dataname2id[dataname]])
  for i in range(len(out)):
    topk_idx = nn_result[i]
    ranking = idx2name[topk_idx].tolist()
    out[i]['ranking'] = ranking
  return out

def main(opt):
  root_dir = opt.root_dir
  split = opt.split
  modelname = opt.model
  img_encoder = opt.img_encoder
  text_encoder = opt.text_encoder
  embed_dim = opt.embed_dim
  model_dir = "{}.{}.{}.{}".format(modelname, img_encoder, text_encoder, embed_dim)

  anno_dir = root_dir + "/annotation"
  split_dir = root_dir + "/public_split"
  test_names = []
  captions = []
  for name in datanames:
    test_names.append(json.load(open("{}/split.{}.{}.json".format(split_dir, name, split))))
    captions.append(json.load(open("{}/ref_caption_{}_{}.json".format(anno_dir, name, split))))

  for name in datanames:
    print(name)
    sims = np.load('results/{}/{}.{}.{}.scores.npy'.format(model_dir, split, name, modelname))
     
    out = convert_sims_to_submit(sims, name, captions, test_names)
    out_dir = 'results/{}/{}'.format(model_dir, split)
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    json.dump(out, open("{}/{}.predict.json".format(out_dir, name), "w"))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--root_dir', type=str, default='./data')
  parser.add_argument('--split', type=str, default='val')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--img_encoder', type=str, default='efficientnet')
  parser.add_argument('--text_encoder', type=str, default='dualenc')
  parser.add_argument('--embed_dim', type=int, default=1024)
  args = parser.parse_args()
  main(args)