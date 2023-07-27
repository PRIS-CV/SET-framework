from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse

# model
import _init_paths
from layers.twophrase_model_strong import TwoPhrase
from loaders.twophrase_dataloader import DataLoader
import evals.twophrase_eval_strong as eval_utils

# torch
import torch
import torch.nn as nn


def load_model(checkpoint_path, opt):
    tic = time.time()
    model = TwoPhrase(opt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time() - tic))
    return model

def evaluate(params):
    # load mode info
    #model_prefix = osp.join('outputV2', params['dataset_splitBy'], params['id'], 'mrcn_cmr_with_st')
    model_prefix = osp.join('twophrase_output','strong' ,params['dataset_splitBy'], params['id'], 'mrcn_cmr_with_st')
    # model_prefix = '/backup/chenyitao/DTWREG-master/output/refcoco+_unc/1/mrcn_cmr_with_st'
    infos = json.load(open(model_prefix + '.json'))
    model_opt = infos['opt']
    model_path = model_prefix + '.pth'
    model = load_model(model_path, model_opt)
    # model_opt['dataset'] = params['dataset']

    # set up loader
    data_json = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/prepro', params['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/prepro', params['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join('cache/sub_obj_wds', model_opt['dataset_splitBy'], 'sub_obj.json')
    # similarity = osp.join('cache/similarity', model_opt['dataset_splitBy'], 'similarity.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, opt=model_opt)

    # loader's feats
    feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
    args.imdb_name = model_opt['imdb_name']
    args.net_name = model_opt['net_name']
    args.tag = model_opt['tag']
    args.iters = model_opt['iters']
    loader.prepare_mrcn(head_feats_dir=osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir),
                        args=args)
    ann_feats = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/feats', model_opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
    # load ann features
    loader.loadFeats({'ann': ann_feats})

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    # evaluate on the split,
    split = params['split']
    model_opt['verbose'] = params['verbose']

    val_loss, acc, predictions = eval_utils.eval_split(loader, model, split, model_opt)

    print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
          (params['dataset_splitBy'], params['split'], len(predictions), acc * 100.))

    # save
    #out_dir = osp.join('resultsV2', params['dataset_splitBy'], 'easy')
    out_dir = osp.join('twophrase_results','strong' ,params['dataset_splitBy'], 'easy')
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = osp.join(out_dir, params['id'] + '_' + params['split'] + '.json')
    with open(out_file, 'w') as of:
        json.dump({'predictions': predictions, 'acc': acc}, of)

    # write to results.txt
    f = open('experiments/easy_results.txt', 'a')
    f.write('[%s]: [%s][%s], id[%s]\'s acc is %.2f%%\n' % \
            (params['id'], params['dataset_splitBy'], params['split'], params['id'], acc * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco',
                        help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default='0', help='model id name')
    parser.add_argument('--verbose', type=int, default=1, help='if we want to print the testing progress')
    args = parser.parse_args()
    params = vars(args)

    # make other options
    params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
    evaluate(params)


