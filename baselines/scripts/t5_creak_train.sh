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