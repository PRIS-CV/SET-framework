"""
data_json has 
0. refs:       [{ref_id, ann_id, box, image_id, split, category_id, sent_ids, att_wds}]
1. images:     [{image_id, ref_ids, file_name, width, height, h5_id}]
2. anns:       [{ann_id, category_id, image_id, box, h5_id}]
3. sentences:  [{sent_id, tokens, h5_id}]
4. word_to_ix: {word: ix}
5. att_to_ix : {att_wd: ix}
6. att_to_cnt: {att_wd: cnt}
7. label_length: L

Note, box in [xywh] format
label_h5 has
/labels is (M, max_length) uint32 array of encoded labels, zeros padded
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import io

import os.path as osp
import numpy as np
import h5py
import json
import random

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
from loaders.loader import Loader
from mrcn import inference_no_imdb
import functools

import json


# box functions
def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class DataLoader(Loader):

    def __init__(self, data_json, data_h5, sub_obj_wds, opt):
        # parent loader instance
        Loader.__init__(self, data_json, sub_obj_wds, data_h5)
        # prepare attributes
        self.att_to_ix = self.info['att_to_ix']
        self.ix_to_att = {ix: wd for wd, ix in self.att_to_ix.items()}
        self.num_atts = len(self.att_to_ix)
        self.att_to_cnt = self.info['att_to_cnt']

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}
        for image_id, image in self.Images.items():
            # we use its ref's split (there is assumption that each image only has one split)
            split = self.Refs[image['ref_ids'][0]]['split']
            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0
            self.split_ix[split] += [image_id]
        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' %(len(v), k))

        # self.w2s_image_id = []
        # self.w2s_iterators = 0
        # self.w2s_sentid2annid = {}
        # pre_result_file='/backup/chenyitao/DTWREG-master/twophrase_results/'+opt['dataset_splitBy']+'/easy/1_train.json'
        # self.predictions=json.load(open(pre_result_file))['predictions']
        # for entry in self.predictions:
        #     if entry['score']>0.0364:
        #         if entry['image_id'] not in self.w2s_image_id:
        #             self.w2s_image_id.append(entry['image_id'])
        #         self.w2s_sentid2annid[entry['sent_id']]=entry['pred_ann_id']

    def prepare_mrcn(self, head_feats_dir, args):
        """
        Arguments:
          head_feats_dir: cache/feats/dataset_splitBy/net_imdb_tag, containing all image conv_net feats
          args: imdb_name, net_name, iters, tag
        """
        self.head_feats_dir = head_feats_dir
        self.mrcn = inference_no_imdb.Inference(args)
        assert args.net_name == 'res101'
        self.pool5_dim = 1024
        self.fc7_dim = 2048

    # load different kinds of feats
    def loadFeats(self, Feats):
        # Feats = {feats_name: feats_path}
        self.feats = {}
        self.feat_dim = None
        for feats_name, feats_path in Feats.items():
            if osp.isfile(feats_path):
                self.feats[feats_name] = h5py.File(feats_path, 'r')
                self.feat_dim = self.feats[feats_name]['fc7'].shape[1]
                assert self.feat_dim == self.fc7_dim
                print('FeatLoader loading [%s] from %s [feat_dim %s]' %(feats_name, feats_path, self.feat_dim))

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])
    
    def w2s_shuffle(self):
        random.shuffle(self.w2s_image_id)

    # reset iterator
    def resetIterator(self, split):
        self.iterators[split]=0

    # expand list by seq per ref, i.e., [a,b], 3 -> [aaabbb]
    def expand_list(self, L, n):
        out = []
        for l in L:
            out += [l] * n
        return out

    def image_to_head(self, image_id):
        """Returns
        head: float32 (1, 1024, H, W)
        im_info: float32 [[im_h, im_w, im_scale]]
        """
        feats_h5 = osp.join(self.head_feats_dir, str(image_id)+'.h5')
        feats = h5py.File(feats_h5, 'r')
        head, im_info = feats['head'], feats['im_info']
        return np.array(head), np.array(im_info)

    def fetch_neighbour_ids(self, ann_id):
        """
        For a given ann_id, we return
        - st_ann_ids: same-type neighbouring ann_ids (not include itself)
        - dt_ann_ids: different-type neighbouring ann_ids
        Ordered by distance to the input ann_id
        """
        ann = self.Anns[ann_id]
        x,y,w,h = ann['box']
        rx, ry = x+w/2, y+h/2

        @functools.cmp_to_key
        def compare(ann_id0, ann_id1):
            x,y,w,h = self.Anns[ann_id0]['box']
            ax0, ay0 = x+w/2, y+h/2
            x,y,w,h = self.Anns[ann_id1]['box']
            ax1, ay1 = x+w/2, y+h/2
            # closer to farmer
            if (rx-ax0)**2+(ry-ay0)**2 <= (rx-ax1)**2+(ry-ay1)**2:
                return -1
            else:
                return 1

        image = self.Images[ann['image_id']]

        ann_ids = list(image['ann_ids'])
        ann_ids = sorted(ann_ids, key=compare)

        st_ann_ids, dt_ann_ids = [], []
        for ann_id_else in ann_ids:
            if ann_id_else != ann_id:
                if self.Anns[ann_id_else]['category_id'] == ann['category_id']:
                    st_ann_ids += [ann_id_else]
                else:
                    dt_ann_ids +=[ann_id_else]
        return st_ann_ids, dt_ann_ids

    def fetch_grid_feats(self, boxes, net_conv, im_info):
        """returns -pool5 (n, 1024, 7, 7) -fc7 (n, 2048, 7, 7)"""
        pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)
        return pool5, fc7

    def compute_lfeats(self, ann_ids):
        # return ndarray float32 (#ann_ids, 5)
        lfeats = np.empty((len(ann_ids), 5), dtype=np.float32)
        for ix, ann_id in enumerate(ann_ids):
            ann = self.Anns[ann_id]
            image = self.Images[ann['image_id']]
            x, y ,w, h = ann['box']
            ih, iw = image['height'], image['width']
            lfeats[ix] = np.array([x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)],np.float32)
        return lfeats

    def compute_dif_lfeats(self, ann_ids, topK=5):
        # return ndarray float32 (#ann_ids, 5*topK)
        dif_lfeats = np.zeros((len(ann_ids), 5*topK), dtype=np.float32)
        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx,rcy,rw,rh = rbox[0]+rbox[2]/2,rbox[1]+rbox[3]/2,rbox[2],rbox[3]
            st_ann_ids, _ =self.fetch_neighbour_ids(ann_id)
            # candidate box
            for j, cand_ann_id in enumerate(st_ann_ids[:topK]):
                cbox = self.Anns[cand_ann_id]['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                dif_lfeats[i, j*5:(j+1)*5] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return dif_lfeats


    def fetch_attribute_label(self, ref_ids):
        """Return
    - labels    : Variable float (N, num_atts)
    - select_ixs: Variable long (n, )
    """
        labels = np.zeros((len(ref_ids), self.num_atts))
        select_ixs = []
        for i, ref_id in enumerate(ref_ids):
            # pdb.set_trace()
            ref = self.Refs[ref_id]
            if len(ref['att_wds']) > 0:
                select_ixs += [i]
                for wd in ref['att_wds']:
                    labels[i, self.att_to_ix[wd]] = 1

        return Variable(torch.from_numpy(labels).float().cuda()), Variable(torch.LongTensor(select_ixs).cuda())

    def fetch_cxt_feats(self, ann_ids, opt):
        """
        Return
        - cxt_feats : ndarray (#ann_ids, fc7_dim)
        - cxt_lfeats: ndarray (#ann_ids, ann_ids, 5)
        - dist: ndarray (#ann_ids, ann_ids, 1)
        Note we only use neighbouring "different" (+ "same") objects for computing context objects, zeros padded.
        """

        cxt_feats = np.zeros((len(ann_ids), self.fc7_dim), dtype=np.float32)
        cxt_lfeats = np.zeros((len(ann_ids), len(ann_ids), 5), dtype=np.float32)
        dist = np.zeros((len(ann_ids), len(ann_ids), 1), dtype=np.float32)

        for i, ann_id in enumerate(ann_ids):
            # reference box
            rbox = self.Anns[ann_id]['box']
            rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
            for j, cand_ann_id in enumerate(ann_ids):
                cand_ann = self.Anns[cand_ann_id]
                # fc7_feats
                cxt_feats[i, :] = self.feats['ann']['fc7'][cand_ann['h5_id'], :]
                cbox = cand_ann['box']
                cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
                ccx, ccy = cbox[0]+cbox[2]/2, cbox[1]+cbox[3]/2
                dist[i,j,:] = np.array(abs(rcx-ccx)+abs(rcy-ccy))
                cxt_lfeats[i,j,:] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        return cxt_feats, cxt_lfeats, dist

    def extract_ann_features(self, image_id, opt):
        """Get features for all ann_ids in an image"""
        image = self.Images[image_id]
        ann_ids = image['ann_ids']

        # fetch image features
        head, im_info = self.image_to_head(image_id)
        head = Variable(torch.from_numpy(head).cuda())

        # fetch ann features
        ann_boxes = xywh_to_xyxy(np.vstack([self.Anns[ann_id]['box'] for ann_id in ann_ids]))
        labelids = [self.Anns[ann_id]['category_id'] for ann_id in ann_ids]
        labelnames = [self.ix_to_cat[labelid] for labelid in labelids]
        labelids=[]
        for name in labelnames:
            wordids=[]
            for word in name.split():
                wordid=self.word_to_ix[word]
                wordids=wordids+[wordid]
            wordids=Variable((torch.Tensor(wordids).cuda()).long())
            labelids=labelids+[wordids]
        ann_pool5, ann_fc7 = self.fetch_grid_feats(ann_boxes, head, im_info)

        # absolute location features
        lfeats = self.compute_lfeats(ann_ids)
        lfeats = Variable(torch.from_numpy(lfeats).cuda())

        # # relative location features
        # dif_lfeats = self.compute_dif_lfeats(ann_ids)
        # dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())

        # # fetch context_fc7 and context_lfeats
        # cxt_fc7, cxt_lfeats, dist = self.fetch_cxt_feats(ann_ids, opt)
        # cxt_fc7 = Variable(torch.from_numpy(cxt_fc7).cuda())
        # cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())
        # dist = Variable(torch.from_numpy(dist).cuda())

        return labelids,ann_fc7, ann_pool5, lfeats#, dif_lfeats, cxt_fc7, cxt_lfeats, dist


    def compute_rel_lfeats(self, sub_ann_id, obj_ann_id):
        if sub_ann_id == -1 or obj_ann_id == -1:
            rel_lfeats = torch.zeros(5)
        else:
            rbox = self.Anns[sub_ann_id]['box']
            rcx, rcy, rw, rh = rbox[0] + rbox[2] / 2, rbox[1] + rbox[3] / 2, rbox[2], rbox[3]
            cbox = self.Anns[obj_ann_id]['box']
            cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
            rel_lfeats = np.array(
                [(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw, (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            rel_lfeats = torch.Tensor(rel_lfeats)
        return rel_lfeats

    # get batch of data
    def getBatch_weak(self, split, opt):
        # options
        batch_size = opt.get('batch_size', 5)
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]


        # fetch image_ids
        batch_image_ids = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            image_id = split_ix[ri]
            batch_image_ids += [image_id]

        ############# load new file #########################

        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        #sent_sub_nounid = sent_extract["sent_sub_nounid"]
        #sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        #sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]
        #sent_rel_prepid = sent_extract["sent_rel_prepid"]



        #####################################################
        batch_ann_fc7=[]
        batch_labelids=[]
        # batch_ann_pool5=[]
        batch_ann_fleats=[]
        batch_sub_wordids = []
        batch_sub_classwordids = []
        batch_obj_wordids = []
        batch_rel_wordids = []

        for image_id in batch_image_ids:
            ann_ids = self.Images[image_id]['ann_ids']
            ann_num = len(ann_ids)
            ref_ids = self.Images[image_id]['ref_ids']
            labelids,ann_fc7, ann_pool5, ann_fleats = self.extract_ann_features(image_id, opt)
            # labelids=(torch.Tensor(np.array(labelids)).cuda()).long()
            ann_fc7.detach_()
            ann_pool5.detach_()
            batch_labelids=batch_labelids+[labelids]
            batch_ann_fc7=batch_ann_fc7+[ann_fc7]
            # batch_ann_pool5=batch_ann_pool5+[ann_pool5]
            batch_ann_fleats=batch_ann_fleats+[ann_fleats]

            sub_wordids = []
            sub_classwordids = []
            obj_wordids = []
            rel_wordids = []

            img_sent_ids = []
            for ref_id in ref_ids:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    img_sent_ids += [sent_id]
            if len(img_sent_ids)==1:
                img_sent_ids=img_sent_ids*2
            random.shuffle(img_sent_ids)
            for sent_id in img_sent_ids:

                sub_wordid = sent_sub_wordid[str(sent_id)]
                sub_wordids.append(sub_wordid)

                sub_classwordid = sent_sub_classwordid[str(sent_id)]
                sub_classwordids.append(sub_classwordid)

                obj_wordid = sent_obj_wordid[str(sent_id)]
                obj_wordids.append(obj_wordid)

                rel_wordid = sent_rel_wordid[str(sent_id)]
                rel_wordids.append(rel_wordid)
            
            #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
            sub_wordids = Variable((torch.Tensor(np.array(sub_wordids)).cuda()).long())

            #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
            obj_wordids = Variable((torch.Tensor(np.array(obj_wordids)).cuda()).long())

            #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
            rel_wordids = Variable((torch.Tensor(np.array(rel_wordids)).cuda()).long())

            #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
            sub_classwordids = Variable((torch.Tensor(np.array(sub_classwordids)).cuda()).long())

            batch_sub_wordids = batch_sub_wordids+[sub_wordids]
            batch_sub_classwordids = batch_sub_classwordids+[sub_classwordids]
            batch_obj_wordids = batch_obj_wordids+[obj_wordids]
            batch_rel_wordids = batch_rel_wordids+[rel_wordids]


        data = {}
        #visual:[B,r_N,v_dim]  lang:[B,s_N,l_dim]
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = batch_sub_wordids

        #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = batch_obj_wordids

        #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = batch_rel_wordids
        #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = batch_sub_classwordids
        data['label'] = batch_labelids

        # data['ann_pool5'] = batch_ann_pool5
        data['ann_fc7'] = batch_ann_fc7
        data['ann_fleats'] = batch_ann_fleats

        #########################################################

        return data

    def get_attribute_weights(self, scale = 10):
        # weights = \lamda * 1/sqrt(cnt)
        cnts = [self.att_to_cnt[self.ix_to_att[ix]] for ix in range(self.num_atts)]
        cnts = np.array(cnts)
        weights = 1 / cnts ** 0.5
        weights = (weights-np.min(weights))/(np.max(weights)-np.min(weights))
        weights = weights * (scale - 1) + 1
        return torch.from_numpy(weights).float()

    def decode_attribute_label(self, scores):
        """- scores: Variable (cuda) (n, num_atts) after sigmoid range [0, 1]
           - labels:list of [[att, sc], [att, sc], ...
        """
        scores = scores.data.cpu().numpy()
        N = scores.shape[0]
        labels = []
        for i in range(N):
            label = []
            score = scores[i]
            for j, sc in enumerate(list(score)):
                label += [(self.ix_to_att[j], sc)]
                labels.append(label)
        return labels



    def getTestBatch(self, split, opt):
        # options
        batch_size = opt.get('batch_size', 5)
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]

        # fetch image_ids
        batch_image_ids = []
        batch_ann_ids = []
        batch_gd_boxes = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            image_id = split_ix[ri]
            batch_image_ids += [image_id]
            image = self.Images[image_id]
            ann_ids = image['ann_ids']

            sent_ids = []
            gd_ixs = []
            gd_boxes = []
            for ref_id in image['ref_ids']:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    sent_ids += [sent_id]
                    gd_ixs += [ann_ids.index(ref['ann_id'])]
                    gd_boxes += [ref['box']]
            batch_gd_boxes+=[gd_boxes]

            expand_ann_ids = []
            for ann_id in ann_ids:
                for i in range(len(ann_ids)):
                    expand_ann_ids.append(ann_id)
            batch_ann_ids+=[expand_ann_ids]

        ############# load new file #########################

        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]

        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]

        sent_rel_wordid = sent_extract["sent_rel_wordid"]


        #####################################################
        batch_ann_fc7=[]
        batch_labelids=[]
        # batch_ann_pool5=[]
        batch_ann_fleats=[]
        batch_sub_wordids = []
        batch_sub_classwordids = []
        batch_obj_wordids = []
        batch_rel_wordids = []
        batch_sent_ids=[]

        for image_id in batch_image_ids:
            ann_ids = self.Images[image_id]['ann_ids']
            ann_num = len(ann_ids)
            ref_ids = self.Images[image_id]['ref_ids']
            labelids,ann_fc7, ann_pool5, ann_fleats = self.extract_ann_features(image_id, opt)
            # labelids=(torch.Tensor(np.array(labelids)).cuda()).long()
            ann_fc7.detach_()
            ann_pool5.detach_()
            batch_labelids=batch_labelids+[labelids]
            batch_ann_fc7=batch_ann_fc7+[ann_fc7]
            # batch_ann_pool5=batch_ann_pool5+[ann_pool5]
            batch_ann_fleats=batch_ann_fleats+[ann_fleats]

            sub_wordids = []
            sub_classwordids = []
            obj_wordids = []
            rel_wordids = []

            img_sent_ids = []
            for ref_id in ref_ids:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    img_sent_ids += [sent_id]
            batch_sent_ids = batch_sent_ids+[img_sent_ids]
            for sent_id in img_sent_ids:

                sub_wordid = sent_sub_wordid[str(sent_id)]
                sub_wordids.append(sub_wordid)

                sub_classwordid = sent_sub_classwordid[str(sent_id)]
                sub_classwordids.append(sub_classwordid)

                obj_wordid = sent_obj_wordid[str(sent_id)]
                obj_wordids.append(obj_wordid)

                rel_wordid = sent_rel_wordid[str(sent_id)]
                rel_wordids.append(rel_wordid)
            
            #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
            sub_wordids = Variable((torch.Tensor(np.array(sub_wordids)).cuda()).long())
            # sub_wordids.detach_()

            #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
            obj_wordids = Variable((torch.Tensor(np.array(obj_wordids)).cuda()).long())
            # obj_wordids.detach_()

            #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
            rel_wordids = Variable((torch.Tensor(np.array(rel_wordids)).cuda()).long())
            # rel_wordids.detach_()

            #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
            sub_classwordids = Variable((torch.Tensor(np.array(sub_classwordids)).cuda()).long())
            # sub_classwordids.detach_()

            batch_sub_wordids = batch_sub_wordids+[sub_wordids]
            batch_sub_classwordids = batch_sub_classwordids+[sub_classwordids]
            batch_obj_wordids = batch_obj_wordids+[obj_wordids]
            batch_rel_wordids = batch_rel_wordids+[rel_wordids]


        data = {}
        #visual:[B,r_N,v_dim]  lang:[B,s_N,l_dim]
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = batch_sub_wordids

        #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = batch_obj_wordids

        #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = batch_rel_wordids
        #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = batch_sub_classwordids
        data['label'] = batch_labelids

        # data['ann_pool5'] = batch_ann_pool5
        data['ann_fc7'] = batch_ann_fc7
        data['ann_fleats'] = batch_ann_fleats

        data['batch_ann_ids']=batch_ann_ids
        data['batch_gd_boxes']=batch_gd_boxes

        data['batch_image_ids']=batch_image_ids
        data['batch_sent_ids']=batch_sent_ids

        #########################################################

        return data

    def getBatch_strong(self, split, opt):
        # options
        batch_size = opt.get('batch_size', 5)
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]


        # fetch image_ids
        batch_image_ids = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            image_id = split_ix[ri]
            batch_image_ids += [image_id]

        ############# load new file #########################

        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        #sent_sub_nounid = sent_extract["sent_sub_nounid"]
        #sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        #sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]
        #sent_rel_prepid = sent_extract["sent_rel_prepid"]



        #####################################################
        batch_ann_fc7=[]
        batch_labelids=[]
        # batch_ann_pool5=[]
        batch_ann_fleats=[]
        batch_sub_wordids = []
        batch_sub_classwordids = []
        batch_obj_wordids = []
        batch_rel_wordids = []
        batch_psent_to_pann = []

        for image_id in batch_image_ids:
            ann_ids = self.Images[image_id]['ann_ids']
            ann_num = len(ann_ids)
            ref_ids = self.Images[image_id]['ref_ids']
            labelids,ann_fc7, ann_pool5, ann_fleats = self.extract_ann_features(image_id, opt)
            # labelids=(torch.Tensor(np.array(labelids)).cuda()).long()
            ann_fc7.detach_()
            ann_pool5.detach_()
            batch_labelids=batch_labelids+[labelids]
            batch_ann_fc7=batch_ann_fc7+[ann_fc7]
            # batch_ann_pool5=batch_ann_pool5+[ann_pool5]
            batch_ann_fleats=batch_ann_fleats+[ann_fleats]

            sub_wordids = []
            sub_classwordids = []
            obj_wordids = []
            rel_wordids = []
            psent_to_pann=[]
            img_sent_ids = []

            tmp_img_sent_ids=[]
            for ref_id in ref_ids:
                ref = self.Refs[ref_id]
                annid=ann_ids.index(ref['ann_id'])
                for sent_id in ref['sent_ids']:
                    tmp_img_sent_ids += [[sent_id,annid]]
            if len(tmp_img_sent_ids)==1:
                tmp_img_sent_ids=tmp_img_sent_ids*2
            random.shuffle(tmp_img_sent_ids)
            img_sent_ids=[tmp[0] for tmp in tmp_img_sent_ids]
            psent_to_pann=[tmp[1] for tmp in tmp_img_sent_ids]
            batch_psent_to_pann = batch_psent_to_pann+[psent_to_pann]
            
            for sent_id in img_sent_ids:

                sub_wordid = sent_sub_wordid[str(sent_id)]
                sub_wordids.append(sub_wordid)

                sub_classwordid = sent_sub_classwordid[str(sent_id)]
                sub_classwordids.append(sub_classwordid)

                obj_wordid = sent_obj_wordid[str(sent_id)]
                obj_wordids.append(obj_wordid)

                rel_wordid = sent_rel_wordid[str(sent_id)]
                rel_wordids.append(rel_wordid)
            
            #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
            sub_wordids = Variable((torch.Tensor(np.array(sub_wordids)).cuda()).long())

            #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
            obj_wordids = Variable((torch.Tensor(np.array(obj_wordids)).cuda()).long())

            #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
            rel_wordids = Variable((torch.Tensor(np.array(rel_wordids)).cuda()).long())

            #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
            sub_classwordids = Variable((torch.Tensor(np.array(sub_classwordids)).cuda()).long())

            batch_sub_wordids = batch_sub_wordids+[sub_wordids]
            batch_sub_classwordids = batch_sub_classwordids+[sub_classwordids]
            batch_obj_wordids = batch_obj_wordids+[obj_wordids]
            batch_rel_wordids = batch_rel_wordids+[rel_wordids]


        data = {}
        #visual:[B,r_N,v_dim]  lang:[B,s_N,l_dim]
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = batch_sub_wordids

        #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = batch_obj_wordids

        #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = batch_rel_wordids
        #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = batch_sub_classwordids
        data['label'] = batch_labelids

        # data['ann_pool5'] = batch_ann_pool5
        data['ann_fc7'] = batch_ann_fc7
        data['ann_fleats'] = batch_ann_fleats

        data['psent_to_pann'] = batch_psent_to_pann

        #########################################################

        return data

    def getTestBatch_strong(self, split, opt):
        # options
        batch_size = opt.get('batch_size', 5)
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1
        wrapped = False
        TopK = opt['num_cxt']

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]


        # fetch image_ids
        batch_image_ids = []
        batch_gd_boxes = []
        batch_ann_ids = []
        for i in range(batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.iterators[split] = ri_next
            image_id = split_ix[ri]
            batch_image_ids += [image_id]
            image = self.Images[image_id]
            ann_ids = image['ann_ids']

            sent_ids = []
            gd_ixs = []
            gd_boxes = []
            for ref_id in image['ref_ids']:
                ref = self.Refs[ref_id]
                for sent_id in ref['sent_ids']:
                    sent_ids += [sent_id]
                    gd_ixs += [ann_ids.index(ref['ann_id'])]
                    gd_boxes += [ref['box']]
            batch_gd_boxes+=[gd_boxes]

            batch_ann_ids+=[ann_ids]

        ############# load new file #########################

        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        #sent_sub_nounid = sent_extract["sent_sub_nounid"]
        #sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        #sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]
        #sent_rel_prepid = sent_extract["sent_rel_prepid"]



        #####################################################
        batch_ann_fc7=[]
        batch_labelids=[]
        # batch_ann_pool5=[]
        batch_ann_fleats=[]
        batch_sub_wordids = []
        batch_sub_classwordids = []
        batch_obj_wordids = []
        batch_rel_wordids = []
        batch_psent_to_pann = []
        batch_sent_ids = []

        for image_id in batch_image_ids:
            ann_ids = self.Images[image_id]['ann_ids']
            ann_num = len(ann_ids)
            ref_ids = self.Images[image_id]['ref_ids']
            labelids,ann_fc7, ann_pool5, ann_fleats = self.extract_ann_features(image_id, opt)
            # labelids=(torch.Tensor(np.array(labelids)).cuda()).long()
            ann_fc7.detach_()
            ann_pool5.detach_()
            batch_labelids=batch_labelids+[labelids]
            batch_ann_fc7=batch_ann_fc7+[ann_fc7]
            # batch_ann_pool5=batch_ann_pool5+[ann_pool5]
            batch_ann_fleats=batch_ann_fleats+[ann_fleats]

            sub_wordids = []
            sub_classwordids = []
            obj_wordids = []
            rel_wordids = []
            psent_to_pann=[]
            img_sent_ids = []

            tmp_img_sent_ids=[]
            for ref_id in ref_ids:
                ref = self.Refs[ref_id]
                annid=ann_ids.index(ref['ann_id'])
                for sent_id in ref['sent_ids']:
                    tmp_img_sent_ids += [[sent_id,annid]]
            img_sent_ids=[tmp[0] for tmp in tmp_img_sent_ids]
            batch_sent_ids = batch_sent_ids+[img_sent_ids]
            psent_to_pann=[tmp[1] for tmp in tmp_img_sent_ids]
            batch_psent_to_pann = batch_psent_to_pann+[psent_to_pann]
            
            for sent_id in img_sent_ids:

                sub_wordid = sent_sub_wordid[str(sent_id)]
                sub_wordids.append(sub_wordid)

                sub_classwordid = sent_sub_classwordid[str(sent_id)]
                sub_classwordids.append(sub_classwordid)

                obj_wordid = sent_obj_wordid[str(sent_id)]
                obj_wordids.append(obj_wordid)

                rel_wordid = sent_rel_wordid[str(sent_id)]
                rel_wordids.append(rel_wordid)
            
            #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
            sub_wordids = Variable((torch.Tensor(np.array(sub_wordids)).cuda()).long())

            #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
            obj_wordids = Variable((torch.Tensor(np.array(obj_wordids)).cuda()).long())

            #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
            rel_wordids = Variable((torch.Tensor(np.array(rel_wordids)).cuda()).long())

            #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
            sub_classwordids = Variable((torch.Tensor(np.array(sub_classwordids)).cuda()).long())

            batch_sub_wordids = batch_sub_wordids+[sub_wordids]
            batch_sub_classwordids = batch_sub_classwordids+[sub_classwordids]
            batch_obj_wordids = batch_obj_wordids+[obj_wordids]
            batch_rel_wordids = batch_rel_wordids+[rel_wordids]


        data = {}
        #visual:[B,r_N,v_dim]  lang:[B,s_N,l_dim]
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = batch_sub_wordids

        #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = batch_obj_wordids

        #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = batch_rel_wordids
        #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = batch_sub_classwordids
        data['label'] = batch_labelids

        # data['ann_pool5'] = batch_ann_pool5
        data['ann_fc7'] = batch_ann_fc7
        data['ann_fleats'] = batch_ann_fleats

        data['psent_to_pann'] = batch_psent_to_pann

        data['batch_sent_ids']=batch_sent_ids
        data['batch_ann_ids']=batch_ann_ids
        data['batch_gd_boxes']=batch_gd_boxes
        data['batch_image_ids']=batch_image_ids

        #########################################################

        return data

    def getBatch_w2s(self, opt):
        # options
        batch_size = 10#opt.get('batch_size', 5)
        max_index = len(self.w2s_image_id) - 1
        wrapped = False

        dataset = opt["dataset"]
        splitBy = opt["splitBy"]


        # fetch image_ids
        batch_image_ids = []
        for i in range(batch_size):
            ri = self.w2s_iterators
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 0
                wrapped = True
            self.w2s_iterators = ri_next
            image_id = self.w2s_image_id[ri]
            batch_image_ids += [image_id]

        ############# load new file #########################

        sent_extract = json.load(open(osp.join('cache/sub_obj_wds', dataset + '_' + splitBy, "sent_extract.json")))
        sent_sub_wordid = sent_extract["sent_sub_wordid"]
        #sent_sub_nounid = sent_extract["sent_sub_nounid"]
        #sent_sub_classid = sent_extract["sent_sub_classid"]
        sent_sub_classwordid = sent_extract["sent_sub_classwordid"]
        sent_obj_wordid = sent_extract["sent_obj_wordid"]
        #sent_obj_nounid = sent_extract["sent_obj_nounid"]
        sent_rel_wordid = sent_extract["sent_rel_wordid"]
        #sent_rel_prepid = sent_extract["sent_rel_prepid"]



        #####################################################
        batch_ann_fc7=[]
        batch_labelids=[]
        # batch_ann_pool5=[]
        batch_ann_fleats=[]
        batch_sub_wordids = []
        batch_sub_classwordids = []
        batch_obj_wordids = []
        batch_rel_wordids = []
        batch_psent_to_pann = []
        batch_sent_ids = []

        for image_id in batch_image_ids:
            ann_ids = self.Images[image_id]['ann_ids']
            ann_num = len(ann_ids)
            ref_ids = self.Images[image_id]['ref_ids']
            labelids,ann_fc7, ann_pool5, ann_fleats = self.extract_ann_features(image_id, opt)
            # labelids=(torch.Tensor(np.array(labelids)).cuda()).long()
            ann_fc7.detach_()
            ann_pool5.detach_()
            batch_labelids=batch_labelids+[labelids]
            batch_ann_fc7=batch_ann_fc7+[ann_fc7]
            # batch_ann_pool5=batch_ann_pool5+[ann_pool5]
            batch_ann_fleats=batch_ann_fleats+[ann_fleats]

            sub_wordids = []
            sub_classwordids = []
            obj_wordids = []
            rel_wordids = []
            psent_to_pann=[]
            img_sent_ids = []

            # tmp_img_sent_ids=[]
            # for ref_id in ref_ids:
            #     ref = self.Refs[ref_id]
            #     annid=ann_ids.index(ref['ann_id'])
            #     for sent_id in ref['sent_ids']:
            #         tmp_img_sent_ids += [[sent_id,annid]]
            # if len(tmp_img_sent_ids)==1:
            #     tmp_img_sent_ids=tmp_img_sent_ids*2
            # random.shuffle(tmp_img_sent_ids)
            # img_sent_ids=[tmp[0] for tmp in tmp_img_sent_ids]
            # psent_to_pann=[tmp[1] for tmp in tmp_img_sent_ids]
            # batch_psent_to_pann = batch_psent_to_pann+[psent_to_pann]

            tmp_img_sent_ids=[]
            for ref_id in ref_ids:
                ref = self.Refs[ref_id]
                # annid=ann_ids.index(ref['ann_id'])
                for sent_id in ref['sent_ids']:
                    if sent_id in self.w2s_sentid2annid:
                        tmp_img_sent_ids+=[[sent_id,ann_ids.index(self.w2s_sentid2annid[sent_id])]]
                        # tmp_img_sent_ids+=[[sent_id,annid]]
            if len(tmp_img_sent_ids)==1:
                tmp_img_sent_ids=tmp_img_sent_ids*2
            random.shuffle(tmp_img_sent_ids)
            img_sent_ids=[tmp[0] for tmp in tmp_img_sent_ids]
            batch_sent_ids = batch_sent_ids+[img_sent_ids]
            psent_to_pann=[tmp[1] for tmp in tmp_img_sent_ids]
            batch_psent_to_pann = batch_psent_to_pann+[psent_to_pann]
            
            for sent_id in img_sent_ids:

                sub_wordid = sent_sub_wordid[str(sent_id)]
                sub_wordids.append(sub_wordid)

                sub_classwordid = sent_sub_classwordid[str(sent_id)]
                sub_classwordids.append(sub_classwordid)

                obj_wordid = sent_obj_wordid[str(sent_id)]
                obj_wordids.append(obj_wordid)

                rel_wordid = sent_rel_wordid[str(sent_id)]
                rel_wordids.append(rel_wordid)
            
            #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
            sub_wordids = Variable((torch.Tensor(np.array(sub_wordids)).cuda()).long())

            #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
            obj_wordids = Variable((torch.Tensor(np.array(obj_wordids)).cuda()).long())

            #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
            rel_wordids = Variable((torch.Tensor(np.array(rel_wordids)).cuda()).long())

            #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
            sub_classwordids = Variable((torch.Tensor(np.array(sub_classwordids)).cuda()).long())

            batch_sub_wordids = batch_sub_wordids+[sub_wordids]
            batch_sub_classwordids = batch_sub_classwordids+[sub_classwordids]
            batch_obj_wordids = batch_obj_wordids+[obj_wordids]
            batch_rel_wordids = batch_rel_wordids+[rel_wordids]


        data = {}
        #visual:[B,r_N,v_dim]  lang:[B,s_N,l_dim]
        data['bounds'] = {'it_pos_now': self.w2s_iterators, 'it_max': max_index, 'wrapped': wrapped}

        #################### new data ############################

        #data['sub_nounids'] = (torch.Tensor(np.array(sub_nounids)).cuda()).long()
        data['sub_wordids'] = batch_sub_wordids

        #data['obj_nounids'] = (torch.Tensor(np.array(obj_nounids)).cuda()).long()
        data['obj_wordids'] = batch_obj_wordids

        #data['rel_prepids'] = (torch.Tensor(np.array(rel_prepids)).cuda()).long()
        data['rel_wordids'] = batch_rel_wordids
        #data['sub_classids'] = (torch.Tensor(np.array(sub_classids)).cuda()).long()
        data['sub_classwordids'] = batch_sub_classwordids
        data['label'] = batch_labelids

        # data['ann_pool5'] = batch_ann_pool5
        data['ann_fc7'] = batch_ann_fc7
        data['ann_fleats'] = batch_ann_fleats

        data['psent_to_pann'] = batch_psent_to_pann

        data['batch_sent_ids']=batch_sent_ids

        #########################################################

        return data
