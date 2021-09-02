# Running Baselines Models

## Getting Started 

### Dependencies

```bash
$ git clone https://github.com/yasumasaonoe/creak.git
```

This code has been tested with Python 3.7 and the following dependencies:

- `torch==1.7.1`
- `transformers==4.9.2`
- `datasets==1.11.0`
- `scikit-learn==0.24.2`
- `deepspeed==0.5.1` (to run T5)

If you're using a conda environment, please use the following commands:

```bash
$ conda create -n creak python=3.7
$ conda activate creak
$ pip install  [package name]
```

## Datasets
### CREAK

- `data/creak`: This directory contains creak train/dev/test/contrast data files.


### Other Datasets
In our paper, we also perform the zero-shot experiments using [FEVER from KILT](https://github.com/facebookresearch/KILT), [FaVIQ](https://github.com/faviq/faviq), and [FoolMeTwice (FM2)](https://github.com/google-research/fool-me-twice). Run these to get the datasets.   

```bash
# FEVER
$ bash ../data/get_fever.sh 

# FaVIQ
$ bash ../data/get_faviq.sh 

# FM2
$ bash ../data/get_fm2.sh 
```
## Experiments

### File Descriptions

- `baselines/run_classifier.py`: Main script for training and evaluating RoBERTa models, and writing predictions to an output file
- `baselines/run_seq2seq_classifier.py`: Main script for training and evaluating T5 models, and writing predictions to an output file

### Train

#### Train RoBERTa on CREAK

To train RoBERTa on CREAK, run this command in the root directory (`creak/`).

```
DATA="creak"
MODEL="roberta-large"
LR="5e-6"
BS="32"
ML="128"
PD_BS="32"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

python baselines/bert_classifiers.py \
  --train_path data/${DATA}/train.json \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/dev.json \
  --contra_path data/${DATA}/contrast_set.json \
  --num_train_epochs 20 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end true \
  --model_name_or_path ${MODEL} \
  --max_seq_length ${ML} \
  --learning_rate ${LR} \
  --per_device_train_batch_size ${PD_BS} \
  --per_device_eval_batch_size 64 \
  --output_dir saved_outputs/${OUTPUT} \
  --do_train \
  --do_eval \
  --do_predict \
  --seed 88888888

```

#### Train T5 on CREAK

```
DATA="creak"
MODEL="t5-3b"
LR="3e-5"
BS="16"
PD_BS="8"
ML="128"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

deepspeed --num_gpus 2 baselines/run_seq2seq_classifiers.py \
  --deepspeed baselines/deepspeed_config.json \
  --train_path data/${DATA}/train.json \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/dev.json \
  --contra_path data/${DATA}/dev.json \
  --num_train_epochs 20 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end \
  --model_name_or_path ${MODEL} \
  --learning_rate ${LR} \
  --per_device_train_batch_size ${PD_BS} \
  --per_device_eval_batch_size ${PD_BS} \
  --output_dir saved_outputs/${OUTPUT} \
  --logging_steps 1 \
  --logging_strategy steps \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --fp16 \
  --seed 88888888
```



### Evaluation
#### Evaluate RoBERTa on CREAK Dev and contrast set
To evaluate RoBERTa on CREAK, run this command in the root directory (`creak/`).
```
CHECKPOINT="XXXX"
DATA="creak"
MODEL="roberta-large"
LR="5e-6"
BS="32"
ML="128"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

python baselines/bert_classifiers.py \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/dev.json \
  --contra_path data/${DATA}/contrast_set.json \
  --model_name_or_path saved_outputs/${OUTPUT}/checkpoint-${CHECKPOINT} \
  --per_device_eval_batch_size 64 \
  --output_dir saved_outputs/${OUTPUT} \
  --do_eval \
  --do_predict \
  --seed 88888888
```

#### Evaluate T5 on CREAK Dev and contrast set
```
CHECKPOINT="XXXX"
DATA="creak"
MODEL="t5-3b"
LR="3e-5"
BS="16"
PD_BS="8"
ML="128"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

deepspeed --num_gpus 2 baselines/run_seq2seq_classifiers.py \
  --deepspeed baselines/deepspeed_config.json \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/test_without_labels.json \
  --contra_path data/${DATA}/contrast_set.json \
  --model_name_or_path saved_outputs/${OUTPUT}/checkpoint-${CHECKPOINT} \
  --per_device_eval_batch_size ${PD_BS} \
  --output_dir saved_outputs/${OUTPUT} \
  --do_eval \
  --do_predict \
  --predict_with_generate \
  --fp16 \
  --seed 88888888
```