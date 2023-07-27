#learning_rate = 0.00001
# learning_rate = 1.26e-5
learning_rate = 1.8e-4      #refcoco:1.26e-4 - 2.26e-4   refcoco+:1.26e-5 - 2.26e-4
max_iters = 7000
# eval_every = 10000
eval_every = 100

# learning_rate_decay_start = 30000
# learning_rate_decay_every = 150000

learning_rate_decay_start = 500
learning_rate_decay_every = 30000

#pair_feat_size = 7168 # pool5 + fc7 + lfeats
#pair_feat_size = 3072 # pool5 + lfeats
pair_feat_size = 5120 # fc7 + fleats

word_emb_size = 300
class_size = 80
noun_candidate_size = 200
prep_candidate_size = 50


