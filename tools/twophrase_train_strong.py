from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import json
import time
import random
import sys

# model
import _init_paths

from loaders.twophrase_dataloader import DataLoader
from layers.twophrase_model_strong import TwoPhrase
import evals.utils as model_utils
import evals.twophrase_eval_strong as eval_utils
from opt import parse_opt
from Config import *

import torch
import pdb

# import nni

def main(args):
    opt = vars(args)
    # initialize
    opt['dataset_splitBy'] = opt['dataset'] + '_' + opt['splitBy']
    checkpoint_dir = osp.join(opt['checkpoint_path'],'strong', opt['dataset_splitBy'], opt['exp_id'])
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    opt['learning_rate'] = 1.0e-4        #coco+:9.346e-5~5e-3    coco:3.746e-4~4e-3
    r1=4e-3#5e-3#4e-3
    r2=4e-4#1e-4#4e-4
    # if opt['dataset']=='refcoco+':
    #     opt['learning_rate'] = learning_rate*0.001
    opt['eval_every'] = 100
    opt['learning_rate_decay_start'] = 500
    opt['learning_rate_decay_every'] = 30000
    opt['pair_feat_size'] = pair_feat_size
    opt['word_emb_size'] = word_emb_size
    opt['class_size'] = class_size
    opt['noun_candidate_size'] = noun_candidate_size
    opt['prep_candidate_size'] = prep_candidate_size
    opt['max_iters'] = 2000
    opt['max_epochs'] = 20


    # set random seed
    torch.manual_seed(opt['seed'])
    random.seed(opt['seed'])

    # set up loader
    data_json = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/prepro', opt['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/prepro', opt['dataset_splitBy'], 'data.h5')
    sub_obj_wds = osp.join('cache/sub_obj_wds', opt['dataset_splitBy'], 'sub_obj.json')
    loader = DataLoader(data_h5=data_h5, data_json=data_json, sub_obj_wds=sub_obj_wds, opt=opt)

    # prepare feats
    feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
    head_feats_dir = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/feats/', opt['dataset_splitBy'], 'mrcn', feats_dir)

    loader.prepare_mrcn(head_feats_dir, args)

    ann_feats = osp.join('/backup/chenyitao/CM-Erase/CM-Erase-REG-master/cache/feats', opt['dataset_splitBy'], 'mrcn',
                         '%s_%s_%s_ann_feats.h5' % (opt['net_name'], opt['imdb_name'], opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    opt['fc7_dim'] = loader.fc7_dim
    opt['vocab_size']= loader.vocab_size
    
    model=TwoPhrase(opt)

    infos = {}
    if opt['start_from'] is not None:
        pass
    iter = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_accuracies = infos.get('val_accuracies', [])
    val_loss_history = infos.get('val_loss_history', {})
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    if opt['load_best_score'] == 1:
        best_val_score = infos.get('best_val_score', None)
    # best_val_score = json.load(open('/backup/chenyitao/DTWREG-master/twophrase_output/'+opt['dataset_splitBy']+'/1/mrcn_cmr_with_st.json'))['best_val_score']
    best_iter=0

    weak_checkpoint = torch.load('/backup/chenyitao/DTWREG-master/twophrase_output/'+opt['dataset_splitBy']+'/1/mrcn_cmr_with_st.pth')
    model.load_state_dict(weak_checkpoint['model'].state_dict())

    # # noisy
    # noisy_checkpoint = torch.load('/backup/chenyitao/DTWREG-master/twophrase_output/strong/'+opt['dataset_splitBy']+'/8/mrcn_cmr_with_st.pth')
    # model.load_state_dict(noisy_checkpoint['model'].state_dict())
    # print('load noisy model done!')

    model.cuda()

    # noisy
    lr = opt['learning_rate']
    # t=(1+(epoch-1)%opt['max_epochs'])/opt['max_epochs']
    # lr=(1-t)*r1+t*r2

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(opt['optim_alpha'], opt['optim_beta']),
                                 eps=opt['optim_epsilon'],
                                 weight_decay=opt['weight_decay'])

    data_time, model_time = 0, 0
    start_time = time.time()

    loader.w2s_shuffle()

    loss_dict={}
    epoch_loss_dict={}

    while True:
        torch.cuda.empty_cache()

        model.train()
        optimizer.zero_grad()

        T = {}

        tic = time.time()
        data=loader.getBatch_w2s(opt)
        sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann, batch_sent_ids=\
            data['sub_wordids'],data['sub_classwordids'],data['obj_wordids'],data['rel_wordids'],data['label'],data['ann_fc7'],data['ann_fleats'],data['psent_to_pann'],data['batch_sent_ids']
        
        T['data'] = time.time() - tic
        tic = time.time()

        # loss,sub_loss,rel_loss,obj_loss,contra_loss,batch_final_att=\
        #     model(sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann)
        loss, batch_final_att=model(sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann)
        # # noisy
        # loss, batch_final_att, batch_loss_record=model(sub_wordids, sub_classids, obj_wordids, rel_wordids,labelids, ann_fc7, ann_fleats, psent_to_pann)
        # batch_loss_dict={}
        # for i in range(len(batch_sent_ids)):
        #     for j in range(len(batch_sent_ids[i])):
        #         batch_loss_dict[batch_sent_ids[i][j]]=batch_loss_record[i][j].item()
        # epoch_loss_dict.update(batch_loss_dict)

        loss.backward()

        model_utils.clip_gradient(optimizer, opt['grad_clip'])
        optimizer.step()

        T['model'] = time.time() - tic
        wrapped = data['bounds']['wrapped']

        data_time += T['data']
        model_time += T['model']

        total_time = (time.time() - start_time)/3600
        total_time = round(total_time, 2)

        print('i[%s], e[%s],loss=%.3f, lr=%.2E, time=%.3f h' % (iter, epoch, loss.data[0].item(), lr, total_time))

        # if iter % opt['losses_log_every'] == 0:
        #     loss_history[iter] = (loss.data[0]).item()

        #     '''
        #     print('iter[%s](epoch[%s]), train_loss=%.3f, lr=%.2E, data:%.2fs/iter, model:%.2fs/iter' \
        #           % (iter, epoch, loss.data[0].item(), lr, data_time / opt['losses_log_every'],
        #              model_time / opt['losses_log_every']) + ", total_time:" + str(total_time) + " h")
        #     '''

        #     #print('i[%s], e[%s], sub_loss=%.1f, obj_loss=%.1f, rel_loss=%.1f, lr=%.2E, time=%.3f h' % (iter, epoch, sub_loss.data[0].item(), obj_loss.data[0].item(), rel_loss.data[0].item(), lr, total_time))
        #     print('i[%s], e[%s], sub_loss=%.3f, obj_loss=%.3f, rel_loss=%.3f, contra_loss=%.3f, lr=%.2E, time=%.3f h' % (iter, epoch, sub_loss.data[0].item(), obj_loss.data[0].item(), rel_loss.data[0].item(), contra_loss.data[0].item(), lr, total_time))

        #     data_time, model_time = 0, 0

        # noisy
        if opt['learning_rate_decay_start'] > 0 and iter > opt['learning_rate_decay_start']:
            frac = (iter - opt['learning_rate_decay_start']) / opt['learning_rate_decay_every']
            decay_factor = 0.1 ** frac
            lr = opt['learning_rate'] * decay_factor
            model_utils.set_lr(optimizer, lr)


        # noisy
        if (iter % opt['eval_every'] == 0) and (iter > 0) or iter == opt['max_iters']:
        # if wrapped:
        #if (iter % opt['eval_every'] == 0) or iter == opt['max_iters']:

            val_loss, acc, predictions = eval_utils.eval_split(loader, model, 'val', opt)
            val_loss_history[iter] = val_loss
            val_result_history[iter] = {'loss': val_loss, 'accuracy': acc}
            val_accuracies += [(iter, acc)]
            print('validation loss: %.2f' % val_loss)
            print('validation acc : %.2f%%\n' % (acc * 100.0))

            current_score = acc

            f = open("./result", "a")
            f.write(str(current_score) + "\n")
            f.close()

            if best_val_score is None or current_score > best_val_score:
                best_iter=iter
                best_val_score = current_score
                best_predictions = predictions
                checkpoint_path = osp.join(checkpoint_dir, opt['id'] + '.pth')
                checkpoint = {}
                checkpoint['model'] = model
                checkpoint['opt'] = opt
                torch.save(checkpoint, checkpoint_path)
                print('model saved to %s' % checkpoint_path)


            # write json report
            infos['iter'] = iter
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['loss_history'] = loss_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_iter'] = best_iter
            infos['best_predictions'] = predictions if best_predictions is None else best_predictions

            infos['opt'] = opt
            infos['val_result_history'] = val_result_history

            #with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w', encoding="utf8") as io:
            # json.dump(infos, io)
            with open(osp.join(checkpoint_dir, opt['id'] + '.json'), 'w') as io:
                json.dump(infos, io)

        iter += 1
        # noisy
        if wrapped:
            loader.w2s_shuffle()
            # loss_dict[epoch]={'lr':lr,'loss':epoch_loss_dict}
            # epoch_loss_dict={}
            # with open(osp.join(checkpoint_dir, 'loss_dict.json'), 'w') as io:
            #     json.dump(loss_dict, io)
            # t=(1+(epoch-1)%opt['max_epochs'])/opt['max_epochs']
            # lr=(1-t)*r1+t*r2
            # model_utils.set_lr(optimizer, lr)
            epoch += 1
        # noisy
        # if epoch>opt['max_epochs']:
        #     break
        if iter >= opt['max_iters'] and opt['max_iters'] > 0:
            print(str(best_val_score))
            print(str(best_iter))
            # nni.report_final_result(best_val_score)
            break

if __name__ == '__main__':
    args = parse_opt()
    # params = nni.get_next_parameter()
    # args = parse_opt(params)
    
    main(args)