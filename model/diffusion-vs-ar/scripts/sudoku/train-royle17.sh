#!/bin/bash
# 训练脚本 for Royle17 数据集

set -e

# 禁用wandb
export WANDB_DISABLED=true

# 创建输出目录
exp=output/sudoku/royle17-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

echo "输出目录: $exp"
echo "使用model_config_tiny模型"
echo "在Royle17上训练"

# 运行训练
echo "开始训练..."
echo "训练日志: $exp/train.log"

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 CUDA_VISIBLE_DEVICES=0 \
accelerate launch --num_machines 1 --mixed_precision fp16 --num_processes 1 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path ../../model/diffusion-vs-ar/output/sudoku/mdm-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-20260222-105609 \
    --do_train \
    --dataset royle17_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 100.0 \
    --plot_loss \
    --run_name sudoku_royle17 \
    --preprocessing_num_workers 8 \
    --fp16 \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --diffusion_steps 20 \
    --save_safetensors False \
    --token_reweighting True \
    --time_reweighting linear \
    --topk_decoding True \
    --alpha 0.25 \
    --gamma 1 \
    > $exp/train.log 2>&1

echo "训练完成"

# 评估
echo "开始评估..."

mkdir -p $exp/royle17_test
echo "评估数据集: royle17_test"

CUDA_VISIBLE_DEVICES=0  \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path ../../model/diffusion-vs-ar/output/sudoku/mdm-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-20260222-105609 \
    --do_predict \
    --cutoff_len 164 \
    --dataset royle17_test \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/royle17_test \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    > $exp/royle17_test/eval.log 2>&1

echo "评估完成: royle17_test"
