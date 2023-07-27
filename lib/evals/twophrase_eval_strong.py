from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math



# IoU function
def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union


def eval_split(loader, model, split, opt):

    verbose = opt.get('verbose', True)
    # assert split != 'train', 'Check the evaluation split.'

    model.eval()

    loader.resetIterator(split)
    loss_sum = 0
    loss_evals = 0
    acc = 0
    acc_evals = 0
    predictions = []
    
    # corr_diff_scores=[]
    # error_diff_scores=[]
    # corr_max_scores=[]
    # error_max_scores=[]
    # corr_var=[]
    # error_var=[]

    while True:
        data = loader.getTestBatch_strong(split, opt)
        sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, batch_ann_ids, batch_gd_boxes, batch_img_ids, batch_sent_ids, psent_to_pann=\
            data['sub_wordids'],data['sub_classwordids'],data['obj_wordids'],data['rel_wordids'],data['label'],data['ann_fc7'],data['ann_fleats'],data['batch_ann_ids'],data['batch_gd_boxes'],data['batch_image_ids'],data['batch_sent_ids'],data['psent_to_pann']
        
        # noisy
        # loss, batch_final_att, batch_loss_record=model(sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann)
        loss, batch_final_att=model(sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann)
        
        loss_sum += loss.data[0].item()
        loss_evals += 1
        
        batch_num=len(batch_final_att)
        for i in range(batch_num):
            img_scores=batch_final_att[i]           #[sent_num,ann_num*ann_num]
            expand_ann_ids=batch_ann_ids[i]         #[ann_num*ann_num]
            img_gd_boxes=batch_gd_boxes[i]          #[sent_num]
            img_id=batch_img_ids[i]
            img_sent_ids=batch_sent_ids[i]          #[sent_num]
            for j in range(img_scores.size(0)):
                pred_ix = torch.argmax(img_scores[j])
                pred_ann_id = expand_ann_ids[pred_ix]
                pred_box = loader.Anns[pred_ann_id]['box']
                gd_box = img_gd_boxes[j]
                IoU = computeIoU(pred_box, gd_box)
                acc_evals += 1
                if IoU >= 0.5:
                    acc += 1
                
                entry = {}
                entry['image_id'] = img_id
                entry['sent_id'] = img_sent_ids[j]
                entry['gd_box'] = gd_box
                entry['pred_box'] = pred_box
                entry['IoU'] = IoU

                predictions.append(entry)
            
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']

        if verbose:
            print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%, loss %.4f' % \
                  (split, ix0, ix1, acc*100.0/acc_evals, loss.data[0].item()))

        if data['bounds']['wrapped']:
            break
    
    return loss_sum / loss_evals, acc / acc_evals, predictions
