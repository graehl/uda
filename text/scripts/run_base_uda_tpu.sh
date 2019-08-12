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
train_tpu=${TPU_ADDRESS:-node-1}
eval_tpu=${TPU_ADDRESS:-node-2}
max_seq_length=${1:-${MAX_SEQ_LENGTH:-128}}
#512
model_dir=${2:-gs://xlnet-logs/uda/text/ckpt/uda_${max_seq_length}}
init_dir=${3:-pretrained_bert_base}

echo $train_tpu $max_seq_length $model_dir

specargs="--sup_train_data_dir=data/proc_data/IMDB/train_20   --unsup_data_dir=data/proc_data/IMDB/unsup   --eval_data_dir=data/proc_data/IMDB/dev   --bert_config_file=pretrained_models/bert_base/bert_config.json   --vocab_file=pretrained_models/bert_base/vocab.txt    --task_name=IMDB"

python main.py \
  --use_tpu=True \
  --tpu_name=${train_tpu} \
  --do_train=True \
  --do_eval=False \
  --init_checkpoint=$init_dir
  $specargs \
  --model_dir=${model_dir} \
  --max_seq_length=${max_seq_length} \
  --num_train_steps=10000 \
  --learning_rate=2e-05 \
  --train_batch_size=32 \
  --num_warmup_steps=1000 \
  --unsup_ratio=7 \
  --uda_coeff=1 \
  --aug_ops=bt-0.9 \
  --aug_copy=1 \
  --uda_softmax_temp=0.85 \
  --tsa=linear_schedule \
  "$@"

echo $eval_tpu $max_seq_length $model_dir

python main.py \
  --use_tpu=True \
  --tpu_name=${eval_tpu} \
  --do_train=False \
  --do_eval=True \
  $specargs \
  --model_dir=${model_dir} \
  --max_seq_length=${max_seq_length} \
  --eval_batch_size=8 \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --train_batch_size=32 \
  --num_warmup_steps=300 \
  "$@"

echo $train_tpu $max_seq_length $model_dir
