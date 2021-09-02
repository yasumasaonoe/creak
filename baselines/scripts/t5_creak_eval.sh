CHECKPOINT=""
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