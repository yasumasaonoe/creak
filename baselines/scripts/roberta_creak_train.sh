export CUDA_VISIBLE_DEVICES=0

DATA="creak"
MODEL="roberta-large"
LR="5e-6"
BS="32"
ML="128"
PD_BS="32"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

python baselines/run_classifiers.py \
  --train_path data/${DATA}/train.json \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/dev.json \
  --contra_path data/${DATA}/dev.json \
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