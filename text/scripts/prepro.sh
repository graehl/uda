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
#!/bin/bash
MAX_SEQ_LENGTH=${1:-128}
pre='--max_seq_length='
if [[ ${MAX_SEQ_LENGTH%$pre} != $MAX_SEQ_LENGTH ]] ; then
    MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH#$pre}
fi
bert_vocab_file=${2:-pretrained_models/bert_base/vocab.txt}
data_dir=${3:-data/proc_data_$MAX_SEQ_LENGTH/IMDB}
sup_size=${4:-20}
lc=True

msl=$pre$MAX_SEQ_LENGTH

set -x
if [[ ! -d $data_dir/IMDB/train_$sup_size ]] ; then
# Preprocess supervised training set
    python preprocess.py \
  --do_lower_case=$lc \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=$data_dir/train_$sup_size \
  --data_type=sup \
  --sub_set=train \
  --sup_size=$sup_size \
  --vocab_file=$bert_vocab_file \
  $msl
fi

if [[ ! -d $data_dir/IMDB/dev ]] ; then
# Preprocess test set
python preprocess.py \
  --do_lower_case=$lc \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=$data_dir/dev \
  --data_type=sup \
  --sub_set=dev \
  --vocab_file=$bert_vocab_file \
  $msl
fi


if [[ ! -d $data_dir/IMDB/unsup ]] ; then
# Preprocess unlabeled set
python preprocess.py \
  --do_lower_case=$lc \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=$data_dir/unsup \
  --back_translation_dir=data/back_translation/imdb_back_trans \
  --data_type=unsup \
  --sub_set=unsup_in \
  --aug_ops=bt-0.9 \
  --aug_copy_num=0 \
  --vocab_file=$bert_vocab_file \
  $msl
echo done
fi
