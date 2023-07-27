from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
import json
import argparse
import os.path as osp
import numpy as np
from Config import *

"""
sent_id: the id of each sentence (142210 totally, 127696 with subject)
word_id: the id of each word in the word list (72740 totally)
word_str: the content of each word
sub_id: the id of each candidate subject
sub_vec: the vector of each candidate subject

sentid_wordid: dictionary, sent_id : word_id
subid_wordid: list, sub_id : word_id
subid_subvec: list, sub_id : sub_vec
sentid_subid: dictionary, sent_id : sub_id
"""

forbidden_att = ['none', 'other', 'sorry', 'pic', 'extreme', 'rightest', 'tie', 'leftest', 'hard', 'only',
                 'darkest', 'foremost', 'topmost', 'leftish', 'utmost', 'lemon', 'good', 'hot', 'more', 'least', 'less',
                 'cant', 'only', 'opposite', 'upright', 'lightest', 'single', 'touching', 'bad', 'main', 'remote',
                 '3pm',
                 'same', 'bottom', 'middle']
forbidden_verb = ['none', 'look', 'be', 'see', 'have', 'head', 'show', 'strip', 'get', 'turn', 'wear',
                  'reach', 'get', 'cross', 'turn', 'point', 'take', 'color', 'handle', 'cover', 'blur', 'close', 'say',
                  'go',
                  'dude', 'do', 'let', 'think', 'top', 'head', 'take', 'that', 'say', 'carry', 'man', 'come', 'check',
                  'stuff',
                  'pattern', 'use', 'light', 'follow', 'rest', 'watch', 'make', 'stop', 'arm', 'try', 'want', 'count',
                  'lead',
                  'know', 'mean', 'lap', 'moniter', 'dot', 'set', 'cant', 'serve', 'surround', 'isnt', 'give', 'click']
forbidden_noun = ['none', 'picture', 'pic', 'screen', 'background', 'camera', 'edge', 'standing', 'thing',
                  'holding', 'end', 'view', 'bottom', 'center', 'row', 'piece', 'right', 'left']


def prep_process(att):
  att_divides = att.split('_')

  if 'prep' in att_divides:
    return att_divides[1]
  else:
    return att


# words: list(u'') to word_id:list, list->list
def words2vocab_indices(words, vocab_dict, UNK_IDENTIFIER):
    if isinstance(words, str):
        vocab_indices = [vocab_dict[words] if words in vocab_dict else vocab_dict[UNK_IDENTIFIER]]
    else:
        vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                         for w in words]
    return vocab_indices

# load vocabulary file
def load_vocab_dict_from_file(dict_file):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    # vocab_dict = {words[n]: n for n in range(len(words))}
    vocab_dict = {n:words[n] for n in range(len(words))}
    return vocab_dict

def transform_vocab(r,vocab_list):
    tr_dict={'ricepotato':'ricepotatoes','herman':'hermans', 'basis':'bases', 'album':'albums', 'somebody':'somebodys', 'jellybean':'jellybeans', 'lugga':'luggage',
             'femla':'female','cowpboy':'cowpboys', 'photoguy':'photoguys','pla':'plate', 'shirtjean':'shirtjeans', 'giffa':'giraffe', 'pastrey':'pastry'}
    tr_dict_keys=list(tr_dict.keys())
    if r[-1]=='y':
      tr_r=r[:-1]+'ies'
    elif r[-2:]=='an':
      tr_r=r[:-2]+'en'
    elif r[-2:]=='um':
      tr_r=r[:-2]+'a'
    elif r[-2:]=='us':
      tr_r=r[:-2]+'i'
    else:
      if r[-1]=='s':
        tr_r=r+'es'
      elif r[-2:]=='ch':
        tr_r=r+'es'
      else:
        tr_r=r+'s'
    
    if tr_r in vocab_list:
      return tr_r
    elif r in tr_dict_keys:
      return tr_dict[r]
    else:
      return 0


def get_sub_obj_rel(ix_To_vocab,vocab_dict, refer, params, vocab_list):

    sents = json.load(open(osp.join('pyutils/refer-parser2/cache/parsed_atts',
                                    params['dataset'] + '_' + params['splitBy'], 'sents.json')))

    # cates = json.load(open(osp.join('cache/sub_obj_wds', params['dataset'] + '_' + params['splitBy'], "cates.json")))
    cates={'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'light': 10, 'hydrant': 11, 'stop': 13, 
         'meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 
         'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'ball': 37, 'kite': 38, 'bat': 39, 'glove': 40, 'skateboard': 41, 
         'surfboard': 42, 'racket': 43, 'bottle': 44, 'wineglass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 
         'broccoli': 56, 'carrot': 57, 'hotdog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'plant': 64, 'bed': 65, 'table': 67, 'toilet': 70, 'tv': 72, 
         'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cellphone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 
         'scissors': 87, 'teddybear': 88, 'drier': 89, 'toothbrush': 90}
    ix_to_cates={v:k for k,v in cates.items()}

    sentToRef = refer.sentToRef

    sent_sub_wordid = {}
    sent_obj_wordid = {}
    sent_rel_wordid = {}

    sent_sub_classwordid = {}

    record_obj_id = []

    for sent in sents:
        sent_id = sent['sent_id']
        atts = sent['atts']
        ref_id = sentToRef[sent_id]['ref_id']

        # # cates is generated through the similarity between the subject word and all possible categories
        # sub_classwordid = cates['cates'][str(sent_id)]
        # sub_classword = ix_To_vocab[sub_classwordid]
        # sub_classwordid = vocab_dict[sub_classword]
        #refcocog cates
        cate_id=refer.Refs[ref_id]['category_id']
        sub_classwordid = vocab_dict[ix_to_cates[cate_id]]
        

        r1 = atts['r1'][0]
        r2 = atts['r2'][0]
        r3 = atts['r3'][0]
        r4 = atts['r4'][0]
        r5 = atts['r5'][0]
        r6 = atts['r6'][0]
        r8_list = atts['r8']

        sub_wordid = vocab_dict['<UNK>']
        obj_wordid = vocab_dict['<UNK>']
        rel_wordid = vocab_dict['<UNK>']

        if r1 != 'none':
          if r1 in vocab_dict:
            sub_wordid = vocab_dict[r1]
          else:
            r1=transform_vocab(r1,vocab_list)
            if r1!=0:
              sub_wordid = vocab_dict[r1]

        if r6 != 'none':
          if r6 in vocab_dict:
            obj_wordid = vocab_dict[r6]
          else:
            r6=transform_vocab(r6,vocab_list)
            if r6!=0:
              obj_wordid = vocab_dict[r6]

        if r5 != 'none':
          r5 = prep_process(r5)
          if r5 in vocab_dict:
            rel_wordid = vocab_dict[r5]
          else:
            r5=transform_vocab(r5,vocab_list)
            if r5!=0:
              rel_wordid = vocab_dict[r5]

        elif r4 != 'none':
          r4 = prep_process(r4)
          if r4 in vocab_dict:
            rel_wordid = vocab_dict[r4]
          else:
            r4=transform_vocab(r4,vocab_list)
            if r4!=0:
              rel_wordid = vocab_dict[r4]

        elif r2 != 'none':
          if r2 in vocab_dict:
            rel_wordid = vocab_dict[r2]
          else:
            r2=transform_vocab(r2,vocab_list)
            if r2!=0:
              rel_wordid = vocab_dict[r2]

        elif r3 != 'none':
          if r3 in vocab_dict:
            rel_wordid = vocab_dict[r3]
          else:
            r3=transform_vocab(r3,vocab_list)
            if r3!=0:
              rel_wordid = vocab_dict[r3]

        elif len(r8_list)>0:
          prep_list = ['left', 'right', 'top', 'bottom', 'higher', 'lower', 'front', 'nearest','middle','closer','closest']

          for prep_str in prep_list:
            if prep_str in r8_list:
              if prep_str in vocab_list:
                rel_wordid = vocab_dict[prep_str]
                break

        # only for test
        record_obj_id.append(obj_wordid)

        # if (sub_wordid!=3) and (obj_wordid==3):
        if (sub_wordid!=vocab_dict['<UNK>']) and (obj_wordid==vocab_dict['<UNK>']):
          obj_wordid = vocab_dict['self']

          # if rel_wordid == 3:
          #   rel_wordid = vocab_dict['self']


        sent_sub_classwordid[sent_id] = sub_classwordid
        sent_sub_wordid[sent_id] = sub_wordid
        sent_obj_wordid[sent_id] = obj_wordid
        sent_rel_wordid[sent_id] = rel_wordid

    return sent_sub_wordid, sent_obj_wordid, sent_rel_wordid, sent_sub_classwordid

def order_word(record_word_id):
  order_word_id = []
  order_word_num = []

  for i in range(len(record_word_id)):

    word_id = record_word_id[i]

    if word_id not in order_word_id:
      order_word_id.append(word_id)
      order_word_num.append(0)
    else:
      order_sub_id = order_word_id.index(word_id)
      order_word_num[order_sub_id] = order_word_num[order_sub_id] + 1

  sort_word_id = []
  sort_word_num = []

  for i in range(len(order_word_id)):
    max_num = max(order_word_num)
    max_order_id = order_word_num.index(max_num)
    max_sub_id = order_word_id[max_order_id]

    sort_word_id.append(max_sub_id)
    sort_word_num.append(max_num)

    order_word_id.pop(max_order_id)
    order_word_num.pop(max_order_id)

  return sort_word_id, sort_word_num

# sub_thre is to generate candidate nouns as well as the link to its corresponding wordid
def sub_thre(sort_word_id, sort_word_num, thre=200):
  subid_wordid = sort_word_id[0:thre]
  subid_wordnum = sort_word_num[0:thre]

  return subid_wordid, subid_wordnum

# convert sent_sub_wordid to sent_obj_nounid (or obj)
def wordid_to_nounid(sentid_wordid, subid_wordid, candidate_size):

  sentid_subid = {}

  for sent_id in sentid_wordid:
    word_id = sentid_wordid[sent_id]

    if word_id in subid_wordid:
      sub_id = subid_wordid.index(word_id)
    else:
      sub_id = candidate_size-1
    sentid_subid[sent_id] = sub_id

  return sentid_subid


def generate_str(sort_word_id, vocab_list):
  str_list = []
  for word_id in sort_word_id:
    word_str = vocab_list[word_id]
    str_list.append(word_str)
  return str_list

def main(params):
    # dataset_splitBy
    data_root, dataset, splitBy = params['data_root'], params['dataset'], params['splitBy']

    # mkdir and write json file
    if not osp.isdir(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy)):
        os.makedirs(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy))

    # load refer
    sys.path.insert(0, 'pyutils/refer')
    from refer import REFER
    refer = REFER(data_root, dataset, splitBy)

    vocab_file = 'cache/word_embedding/vocabulary_72700.txt'
    ix_To_vocab = load_vocab_dict_from_file(vocab_file)

    # f = open(vocab_file, "r")
    # vocab_list = f.read().splitlines()
    # f.close()

    prepro = json.load(open('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/prepro/'+params['dataset'] + '_' + params['splitBy']+'/data.json'))
    vocab_dict=prepro['word_to_ix']
    vocab_list=list(vocab_dict.keys())

    sent_sub_wordid, sent_obj_wordid, sent_rel_wordid, sent_sub_classwordid = get_sub_obj_rel(ix_To_vocab,vocab_dict, refer, params,vocab_list)

    json.dump({"sent_sub_wordid":sent_sub_wordid,  "sent_obj_wordid":sent_obj_wordid,  "sent_rel_wordid":sent_rel_wordid, "sent_sub_classwordid": sent_sub_classwordid}, open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json"), 'w'))

    print('related data have been written!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='sub_obj_wds.json', help='output json file')
    parser.add_argument('--data_root', default='/backup/chenyitao/CM-Erase/CM-Erase-REG-master/pyutils/mask-faster-rcnn/data/refer/data', type=str,
                        help='data folder containing images and four datasets.')
    parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
    parser.add_argument('--images_root', default='', help='root location in which images are stored')


    # argparse
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))

    # call main
    main(params)
