export CUDA_VISIBLE_DEVICES=0

CHECKPOINT="1908"
DATA="creak"
MODEL="roberta-large"
LR="5e-6"
BS="32"
ML="128"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${BS}_ml${ML}"

python baselines/run_classifiers.py \
  --dev_path data/${DATA}/dev.json \
  --test_path data/${DATA}/test_without_labels.json \
  --contra_path data/${DATA}/contrast_set.json \
  --model_name_or_path saved_outputs/${OUTPUT}/checkpoint-${CHECKPOINT} \
  --per_device_eval_batch_size 64 \
  --output_dir saved_outputs/${OUTPUT} \
  --do_eval \
  --do_predict \
  --seed 88888888