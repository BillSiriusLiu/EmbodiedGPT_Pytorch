#!/usr/bin/env bash
python -u ./robohusky/train/train.py\
  --model_name_or_path "../7Btiny/" \
  --cache_dir "../cache/"\
  --conv_style "husky" \
  --train_file "train_files/train_file.json" \
  --output_dir "../checkpoints/" \
  --overwrite_output_dir True \
  --run_name "embodied_family" \
  --freeze_vision_model False \
  --freeze_vision_adapter False \
  --freeze_qformer False \
  --freeze_text_model False \
  --preprocessing_num_workers 1 \
  --pad_to_max_length True \
  --fp16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 2e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --deepspeed "zero_stage2_config.json"