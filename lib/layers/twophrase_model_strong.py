from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from layers.lan_enc import RNNEncoder
from layers.lan_dec import RNNDncoder
from layers.vis_enc import LocationEncoder, SubjectEncoder, RelationEncoder, PairEncoder
import pdb

import random

class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        assert isinstance(bottom, Variable), 'bottom must be variable'

        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)   
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled

class PairEncoder(nn.Module):
    def __init__(self, opt):
        super(PairEncoder, self).__init__()
        self.word_vec_size = opt['word_vec_size']

        # location information
        self.jemb_dim = opt['jemb_dim']
        init_norm = opt.get('visual_init_norm', 20)
        self.lfeats_normalizer = Normalize_Scale(5, init_norm)
        self.fc = nn.Linear(5, opt['jemb_dim'])

        # visual information
        self.fc7_dim = 300
        self.fc7_normalizer = Normalize_Scale(self.fc7_dim, opt['visual_init_norm'])

    def forward(self, ann_fc7, ann_fleats):
        ann_num = ann_fc7.size(0)

        ann_fc7 = self.fc7_normalizer(ann_fc7.contiguous())                                         #[ann_num,300]

        expand_1_fc7 = ann_fc7.unsqueeze(1).expand(ann_num, ann_num, self.fc7_dim)
        expand_0_fc7 = ann_fc7.unsqueeze(0).expand(ann_num, ann_num, self.fc7_dim)

        expand_1_fc7 = expand_1_fc7.contiguous().view(-1, self.fc7_dim)
        expand_0_fc7 = expand_0_fc7.contiguous().view(-1, self.fc7_dim)

        ann_fleats = self.lfeats_normalizer(ann_fleats.contiguous().view(-1, 5))
        ann_fleats = self.fc(ann_fleats)

        expand_1_fleats = ann_fleats.unsqueeze(1).expand(ann_num, ann_num, self.jemb_dim)
        expand_0_fleats = ann_fleats.unsqueeze(0).expand(ann_num, ann_num, self.jemb_dim)

        expand_1_fleats = expand_1_fleats.contiguous().view(-1, self.jemb_dim)
        expand_0_fleats = expand_0_fleats.contiguous().view(-1, self.jemb_dim)

        pair_feats = torch.cat([expand_1_fc7, expand_1_fleats, expand_0_fc7, expand_0_fleats], 1)

        return pair_feats, expand_1_fc7,  expand_0_fc7                    #[ann_num * ann_num, 300*4]

class TwoPhrase(nn.Module):
    def __init__(self, opt):
        super(TwoPhrase, self).__init__()
        self.jemb_dim = opt['jemb_dim']
        self.fc7_dim =  opt['fc7_dim']

        # self.re_sub,self.re_rel,self.re_obj=opt['re_sub'],opt['re_rel'],opt['re_obj']
        # self.final_sub,self.final_rel,self.final_obj=opt['final_sub'],opt['final_rel'],opt['final_obj']
        # self.lamda=opt['lamda']
        # self.ce_sub,self.ce_rel,self.ce_obj=opt['ce_sub'],opt['ce_rel'],opt['ce_obj']
        # self.theta1,self.theta2=opt['theta1'],opt['theta2']

        # self.margin=opt['margin']
        self.margin=0.2472
        self.balance={}

        word_emb_path = '/backup/chenyitao/CM-Erase/CM-Erase-REG-master/glove_emb/'+opt['dataset']+'.npy'
        dict_emb = np.load(word_emb_path)
        self.word_embedding = nn.Embedding(opt['vocab_size'], 300)
        assert dict_emb.shape[0]==opt['vocab_size'],'dict_emb和vocab_size不符'
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(dict_emb).float())
        # self.fc7_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])

        self.pair_encoder = PairEncoder(opt)

        self.linear_v2l = nn.Linear(self.fc7_dim, 300)

        self.linear_Kt = nn.Linear(300,200)
        self.linear_Kd = nn.Linear(300*4,600)
        self.linear_Kr = nn.Linear(300,200)
        self.linear_Qt = nn.Linear(300,200)
        self.linear_Qd = nn.Linear(300*3,600)
        self.linear_Qr = nn.Linear(300,200)
        self.linear_Vvt = nn.Linear(300,200)
        self.linear_Vvd = nn.Linear(300*4,600)
        self.linear_Vvr = nn.Linear(300,200)
        self.linear_Vwt = nn.Linear(300,200)
        self.linear_Vwd = nn.Linear(300*3,600)
        self.linear_Vwr = nn.Linear(300,200)

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.visual_emb = nn.Sequential(
            nn.Linear(200, 300),
            #nn.Linear(self.pool5_dim, 1024),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 300),
            #nn.ReLU(),
        )

        self.pair_emb = nn.Sequential(
            nn.Linear(600, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 300),
            #nn.ReLU(),
        )

    def vis_label_fuse(self,labelids,ann_fc7):
    	#input  
    	#label:  一个list，其中每个元素对应一个image
    	#        list[i]: [ann_num]
    	#ann_fc7:  一个list，其中每个元素对应一个image
    	#        ann_fc7[i]: [ann_num,2048,7,7]
        batch_num=len(ann_fc7)
        batch_ann_emb=[]
        for i in range(batch_num):
            img_ann_fc7 = ann_fc7[i]                #[ann_num,2048,7,7]
            img_labels = labelids[i]
            ann_num=len(img_labels)
            label_embs=[]
            for j in range(ann_num):
                label=img_labels[j]
                label_emb=self.word_embedding(label).mean(0)
                label_embs.append(label_emb.unsqueeze(0))
            img_label_emb=torch.cat(label_embs,0)   #[ann_num,300]

            img_ann_fc7 = img_ann_fc7.view(ann_num, self.fc7_dim, -1).contiguous().mean(2)             #[ann_num,2048]

            # img_ann_fc7 = img_ann_fc7.contiguous().view(ann_num, self.fc7_dim, -1)                   #[ann_num,2048,49]
            # img_ann_fc7 = img_ann_fc7.transpose(1, 2).contiguous().view(-1, self.fc7_dim)            #[ann_num*49,2048]
            # img_ann_fc7 = self.fc7_normalizer(img_ann_fc7)                                           #[ann_num*49,2048]
            # img_ann_fc7 = img_ann_fc7.view(ann_num, 49, -1).transpose(1, 2).contiguous().mean(2)     #[ann_num,2048]

            img_ann_emb=self.linear_v2l(img_ann_fc7)    #[ann_num,300]
            img_ann_emb+=img_label_emb  
              
            batch_ann_emb+=[img_ann_emb]

        return batch_ann_emb

    def attention(self,Q,K):
        D=Q.size(1)
        Q=Q.unsqueeze(1)
        att=torch.sum(Q*K,2)
        att=att/torch.sqrt(torch.tensor(D).float().cuda())
        att = F.softmax(att,1)

        return att

    def mm_loss(self,att,img_psent_to_pann):
        #input
        #att:[sent_num,ann_num]  img_psent_to_pann:[sent_num]
        sent_num=att.size(0)
        ann_num=att.size(1)
        mm_loss=torch.tensor(0).cuda().float()
        for i in range(sent_num):
            sent_att=att[i]
            pann_id=img_psent_to_pann[i]
            for j in range(ann_num):
                if j != int(pann_id):
                    mm_loss+=torch.clamp(self.margin + sent_att[j] - sent_att[pann_id], 0)

        return mm_loss/(sent_num*ann_num)
    
    def record_mm_loss(self,att,img_psent_to_pann):
        #input
        #att:[sent_num,ann_num]  img_psent_to_pann:[sent_num]
        sent_num=att.size(0)
        ann_num=att.size(1)
        mm_loss=torch.tensor(0).cuda().float()
        mm_loss_list=[]
        for i in range(sent_num):
            sent_att=att[i]
            pann_id=img_psent_to_pann[i]
            sample_loss=torch.tensor(0).cuda().float()
            for j in range(ann_num):
                if j != int(pann_id):
                    # mm_loss+=torch.clamp(self.margin + sent_att[j] - sent_att[pann_id], 0)
                    sample_loss+=torch.clamp(self.margin + sent_att[j] - sent_att[pann_id], 0)
            mm_loss+=sample_loss
            mm_loss_list.append(sample_loss/ann_num)

        return mm_loss/(sent_num*ann_num),mm_loss_list
    
    def record_ce_loss(self,att,img_psent_to_pann,balance):
        sent_num=att.size(0)
        ce_loss=torch.tensor(0).cuda().float()
        ce_loss_list=[]
        for i in range(sent_num):
            sent_att=att[i].unsqueeze(0)
            pann_id=img_psent_to_pann[i].unsqueeze(0)
            sample_loss=self.ce_loss(sent_att,pann_id)/balance
            ce_loss_list.append(sample_loss)
            ce_loss+=sample_loss

        return ce_loss/sent_num,ce_loss_list

    def forward(self, sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann):
        ann_fc7=self.vis_label_fuse(labelids,ann_fc7)
        batch_num=len(ann_fc7)
        # batch_re_sub_feats=[]
        # batch_re_pair_feats=[]
        # batch_re_obj_feats=[]
        # batch_V_sub_fuseemb=[]
        # batch_V_pair_wordemb=[]
        # batch_V_obj_wordemb=[]
        # batch_sent_num=[0]
        batch_final_att=[]
        batch_loss_record=[]

        # re_sub,re_rel,re_obj=1,1,1
        final_sub,final_rel,final_obj=0.7,0.4,1

        ce_loss=torch.tensor(0).cuda().float()
        mm_loss=torch.tensor(0).cuda().float()
        for i in range(batch_num):
            img_sub_wordemb=self.word_embedding(sub_wordids[i])                             #[sent_num,300]
            img_sub_classemb=self.word_embedding(sub_classids[i])

            img_obj_wordemb=self.word_embedding(obj_wordids[i])
            img_rel_wordemb=self.word_embedding(rel_wordids[i])
            # img_sub_fuseemb=0.1*img_sub_wordemb+0.9*img_sub_classemb
            img_sub_fuseemb=img_sub_wordemb
            img_pair_wordemb = torch.cat([img_sub_fuseemb, img_obj_wordemb, img_rel_wordemb], 1)
            # batch_sent_num.append(batch_sent_num[-1]+img_sub_classemb.size(0))

            pair_feats,expand_1_fc7,expand_0_fc7=self.pair_encoder(ann_fc7[i],ann_fleats[i])            #[ann_num * ann_num, d]
            ann_num = ann_fc7[i].size(0)
            sent_num = img_sub_wordemb.size(0)

            K_expand_1_fc7=self.linear_Kt(expand_1_fc7)                      #[ann_num * ann_num, 200]
            Q_sub_fuseemb=self.linear_Qt(img_sub_fuseemb)                   #[sent_num, 200]
            sub_att=self.attention(Q_sub_fuseemb,K_expand_1_fc7)            #[sent_num,ann_num * ann_num]

            K_pair_feats=self.linear_Kd(pair_feats)                          #[ann_num * ann_num, 600]
            Q_pair_wordemb=self.linear_Qd(img_pair_wordemb)                   #[sent_num, 600]
            pair_att=self.attention(Q_pair_wordemb,K_pair_feats)

            K_expand_0_fc7=self.linear_Kr(expand_0_fc7)                      #[ann_num * ann_num, 200]
            Q_obj_wordemb=self.linear_Qr(img_obj_wordemb)                    #[sent_num, 200]
            obj_att=self.attention(Q_obj_wordemb,K_expand_0_fc7)

            # V_expand_1_fc7=self.linear_Vvt(expand_1_fc7)                    #[ann_num * ann_num, 200]
            # re_sub_feats=torch.matmul(sub_att,V_expand_1_fc7)               #[sent_num, 200]
            # V_sub_fuseemb=self.linear_Vwt(img_sub_fuseemb)                  #[sent_num, 200]

            # V_pair_feats=self.linear_Vvd(pair_feats)
            # re_pair_feats=torch.matmul(pair_att,V_pair_feats)
            # V_pair_wordemb=self.linear_Vwd(img_pair_wordemb)

            # V_expand_0_fc7=self.linear_Vvr(expand_0_fc7)
            # re_obj_feats=torch.matmul(obj_att,V_expand_0_fc7)
            # V_obj_wordemb=self.linear_Vwr(img_obj_wordemb)

            # #reconstruction
            # sub_result = self.visual_emb(re_sub_feats)
            # obj_result = self.visual_emb(re_obj_feats)
            # rel_result = self.pair_emb(re_pair_feats)

            # detach_img_sub_fuseemb=img_sub_fuseemb.detach()
            # detach_img_obj_wordemb=img_obj_wordemb.detach()
            # detach_img_rel_wordemb=img_rel_wordemb.detach()
            # sub_loss = self.mse_loss(sub_result, detach_img_sub_fuseemb)
            # obj_loss = self.mse_loss(obj_result, detach_img_obj_wordemb)
            # rel_loss = self.mse_loss(rel_result, detach_img_rel_wordemb)

            # re_loss = re_loss+self.re_sub*sub_loss+self.re_rel*rel_loss+self.re_obj*obj_loss
            # final_att=self.final_sub*sub_att+self.final_rel*pair_att+self.final_obj*obj_att        #[sent_num,ann_num * ann_num]

            # re_loss = re_loss+re_sub*sub_loss+re_rel*rel_loss+re_obj*obj_loss
            # final_att=final_sub*sub_att+final_rel*pair_att+final_obj*obj_att        #[sent_num,ann_num * ann_num]

            #strong合并算loss
            # final_att=self.final_sub*sub_att+self.final_rel*pair_att+self.final_obj*obj_att        #[sent_num,ann_num * ann_num]
            final_att=final_sub*sub_att+final_rel*pair_att+final_obj*obj_att        #[sent_num,ann_num * ann_num]
            final_att=final_att.view(sent_num,ann_num,ann_num)
            final_att=torch.sum(final_att,dim=2)                                    #[sent_num,ann_num]
            img_psent_to_pann=torch.tensor(psent_to_pann[i]).cuda()
            # ce_loss使用balance
            if ann_num in self.balance:
                balance=self.balance[ann_num]
            else:
                y=torch.ones(1000).long()
                x1=torch.randn([1000,ann_num])
                x2=torch.randn([1000,30])
                balance=self.ce_loss(x1,y)/self.ce_loss(x2,y)
                self.balance[ann_num]=balance
            ce_loss+=self.ce_loss(final_att,img_psent_to_pann)/balance.cuda()
            # ce_loss+=self.ce_loss(final_att,img_psent_to_pann)
            mm_loss+=self.mm_loss(final_att,img_psent_to_pann)

            # #noisy
            # noisy_ce_loss,ce_loss_list=self.record_ce_loss(final_att,img_psent_to_pann,balance.cuda())
            # noisy_mm_loss,mm_loss_list=self.record_mm_loss(final_att,img_psent_to_pann)
            # record_loss_list=[]
            # for i in range(len(ce_loss_list)):
            #     record_loss_list.append(ce_loss_list[i]+mm_loss_list[i])
            # batch_loss_record+=[record_loss_list]

            # #strong分开算loss
            # img_psent_to_pann=torch.tensor(psent_to_pann[i]).cuda()
            # sub_att=sub_att.view(sent_num,ann_num,ann_num)
            # sub_att=torch.sum(sub_att,dim=2)                                    #[sent_num,ann_num]
            # sub_loss=self.ce_loss(sub_att,img_psent_to_pann)
            # pair_att=pair_att.view(sent_num,ann_num,ann_num)
            # pair_att=torch.sum(pair_att,dim=2)                                    #[sent_num,ann_num]
            # rel_loss=self.ce_loss(pair_att,img_psent_to_pann)
            # obj_att=obj_att.view(sent_num,ann_num,ann_num)
            # obj_att=torch.sum(obj_att,dim=2)                                    #[sent_num,ann_num]
            # obj_loss=self.ce_loss(obj_att,img_psent_to_pann)
            # ce_loss+=self.ce_sub*sub_loss+self.ce_rel*rel_loss+self.ce_obj*obj_loss
            # # final_att=final_sub*sub_att+final_rel*pair_att+final_obj*obj_att
            # final_att=self.final_sub*sub_att+self.final_rel*pair_att+self.final_obj*obj_att

            batch_final_att+=[final_att]
            # batch_re_sub_feats+=[re_sub_feats]
            # batch_re_pair_feats+=[re_pair_feats]
            # batch_re_obj_feats+=[re_obj_feats]
            # batch_V_sub_fuseemb+=[V_sub_fuseemb]
            # batch_V_pair_wordemb+=[V_pair_wordemb]
            # batch_V_obj_wordemb+=[V_obj_wordemb]

        # #对比学习
        # batch_re_sub_feats=torch.cat(batch_re_sub_feats,0)
        # batch_re_pair_feats=torch.cat(batch_re_pair_feats,0)
        # batch_re_obj_feats=torch.cat(batch_re_obj_feats,0)
        # batch_V_sub_fuseemb=torch.cat(batch_V_sub_fuseemb,0)
        # batch_V_pair_wordemb=torch.cat(batch_V_pair_wordemb,0)
        # batch_V_obj_wordemb=torch.cat(batch_V_obj_wordemb,0)
        # sub_sim=torch.matmul(batch_V_sub_fuseemb,batch_re_sub_feats.contiguous().transpose(0,1))               #[sent_num,sent_num]
        # pair_sim=torch.matmul(batch_V_pair_wordemb,batch_re_pair_feats.contiguous().transpose(0,1))
        # obj_sim=torch.matmul(batch_V_obj_wordemb,batch_re_obj_feats.contiguous().transpose(0,1))

        # neg_num=3
        # contra_sub=torch.zeros([batch_sent_num[-1],(batch_num-1)*neg_num+1]).cuda()
        # contra_pair=torch.zeros([batch_sent_num[-1],(batch_num-1)*neg_num+1]).cuda()
        # contra_obj=torch.zeros([batch_sent_num[-1],(batch_num-1)*neg_num+1]).cuda()
        # cnt=0
        # for i in range(batch_sent_num[-1]):
        #     if i >= batch_sent_num[cnt+1]:
        #         cnt+=1
        #     contra_select=[]
        #     for j in range(len(batch_sent_num)-1):
        #         tmp=list(range(batch_sent_num[j],batch_sent_num[j+1]))
        #         if len(tmp)<neg_num:
        #             supply_list=tmp*neg_num
        #             tmp+=supply_list[0:neg_num-len(tmp)]
        #         else:
        #             random.shuffle(tmp)
        #         contra_select+=[tmp[0:neg_num]]
        #     del contra_select[cnt]
        #     contra_select_list=[xx for x in contra_select for xx in x]
        #     contra_sub[i][0]=sub_sim[i][i]
        #     contra_pair[i][0]=pair_sim[i][i]
        #     contra_obj[i][0]=obj_sim[i][i]
        #     for k in range((batch_num-1)*neg_num):
        #         contra_sub[i][k+1]=sub_sim[i][contra_select_list[k]]
        #         contra_pair[i][k+1]=pair_sim[i][contra_select_list[k]]
        #         contra_obj[i][k+1]=obj_sim[i][contra_select_list[k]]
        # contra_sub=-torch.log(F.softmax(contra_sub,1))
        # contra_pair=-torch.log(F.softmax(contra_pair,1))
        # contra_obj=-torch.log(F.softmax(contra_obj,1))
        # contra_loss=(torch.sum((contra_sub+contra_pair+contra_obj),dim=0)[0])/((batch_num-1)*neg_num+1)

        # # lamda=0.2
        # # loss= (re_loss+lamda*contra_loss)/batch_num

        # loss= (re_loss+self.lamda*contra_loss)/batch_num
        ce_loss=ce_loss/batch_num

        mm_loss=mm_loss/batch_num
        loss=ce_loss+mm_loss

        # return loss,sub_loss,rel_loss,obj_loss,contra_loss,batch_final_att
        return loss,batch_final_att#,batch_loss_record
            



