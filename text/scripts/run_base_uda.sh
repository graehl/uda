# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
MSL=${1:-128}
base=${2:-pretrained_models/bert_base}
data=${3:-data/proc_data_$MSL/IMDB}
tpu=${4:-False}
batch_size=${5:-8}
train=${6:-1}
unsupr=${7:-2}
if [[ $train = 1 ]] ; then
    train=True
else
    train=False
fi
python main.py \
  --use_tpu=False \
  --do_train=$train \
  --do_eval=True \
  --sup_train_data_dir=$data/train_20 \
  --unsup_data_dir=$data//unsup \
  --eval_data_dir=$data/dev \
  --bert_config_file=$base/bert_config.json \
  --vocab_file=$base/vocab.txt \
  --init_checkpoint=$base/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=ckpt/base_uda \
  --num_train_steps=10000 \
  --learning_rate=2e-05 \
  --num_warmup_steps=1000 \
  --max_seq_length=$MSL \
  --unsup_ratio=$unsupr \
  --tsa=linear_schedule \
  --aug_ops=bt-0.9 \
  --aug_copy=1 \
  --uda_coeff=1 \
  --train_batch_size=$batch_size \
  --eval_batch_size=$batch_size
