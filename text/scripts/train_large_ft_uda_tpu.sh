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
gs_base=${2:-gs://uda-logs/uda/text}
model_dir=${3:-$gs_base/ckpt/uda_ft_tpu_${max_seq_length}}
init_dir=${4:-$gs_base/imdb_bert_ft} #pretrained_bert_base
data_dir=${5:-$gs_base/proc_data/IMDB}
train_batch=${6:-16}
unsup_ratio=${7:-7}
# unsup batch size 7x sup (based on memory usage?)

echo $train_tpu $max_seq_length $model_dir

specargs="--sup_train_data_dir=$data_dir/train_20   --unsup_data_dir=$data_dir/unsup   --eval_data_dir=$data_dir/dev   --bert_config_file=$init_dir/bert_config.json   --vocab_file=$init_dir/vocab.txt    --task_name=IMDB --train_batch_size=$train_batch --max_seq_length=$max_seq_length"

nodeprecated() {
    grep -v eprecat | grep -v 'Instructions for updating' | egrep -v '^Use.*instead'
}

set -e
set -x

python main.py $specargs \
  --use_tpu=True \
  --tpu_name=${train_tpu} \
  --do_train=True \
  --do_eval_along_training=True \
  --verbose=True \
  --task_name=IMDB \
  --model_dir=${model_dir} \
  --num_train_steps=10000 \
  --learning_rate=2e-05 \
  --num_warmup_steps=1000 \
  --unsup_ratio=$unsup_ratio \
  --uda_coeff=1 \
  --aug_ops=bt-0.9 \
  --aug_copy=1 \
  --uda_softmax_temp=0.85 \
  --tsa=linear_schedule 2>&1 | nodeprecated

python main.py $specargs \
  --use_tpu=True \
  --tpu_name=${eval_tpu} \
  --do_train=False \
  --do_eval=True \
  --task_name=IMDB \
  --model_dir=${model_dir} \
  --eval_batch_size=8 \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --num_warmup_steps=300 2>&1 | nodeprecated
