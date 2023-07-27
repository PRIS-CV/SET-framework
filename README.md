## Prerequisites

* Python 3.6
* Pytorch 0.4
* CUDA 9.0

## Installation the environment

Please first refer to [MattNet](https://github.com/insomnia94/MAttNet) to prepare related data. 

## Generate the Triads
```bash
python ./tools/prepro_rel.py --dataset refcoco --splitBy unc
```

## Training
###  First stage
```bash
CUDA_VISIBLE_DEVICES=1 python ./tools/twophrase_train.py --dataset refcoco --splitBy unc --exp_id 1
```
###  second stage
```bash
CUDA_VISIBLE_DEVICES=1 python ./tools/twophrase_train_strong.py --dataset refcoco --splitBy unc --exp_id 1
```

## Evaluation
###  First stage
```bash
CUDA_VISIBLE_DEVICES=1 python ./tools/twophrase_eval.py --dataset refcoco --splitBy unc --split val --id 1
```
###  seconda stage
```bash
CUDA_VISIBLE_DEVICES=1 python ./tools/twophrase_eval_strong.py --dataset refcoco --splitBy unc --split val --id 1
```
