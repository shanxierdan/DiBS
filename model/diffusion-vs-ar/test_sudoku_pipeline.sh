#!/bin/bash
# 快速测试数独推理 pipeline
# 使用 tiny 模型配置，训练少量 steps，然后评估

set -e

export WANDB_DISABLED=true

# 使用时间戳创建输出目录
exp="test_output/sudoku/mdm-test-$(date +%Y%m%d-%H%M%S)"
mkdir -p $exp

echo "输出目录: $exp"

# 步骤1: 训练少量 steps (10 steps)
echo "=== 步骤1: 训练模型 (10 steps) ==="
CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset sudoku_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 20 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy steps \
    --eval_steps 5 \
    --save_steps 10 \
    --learning_rate 1e-3 \
    --max_steps 10 \
    --max_samples 200 \
    --plot_loss \
    --run_name sudoku_test \
    --preprocessing_num_workers 4 \
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

echo "训练完成，检查点保存在: $exp"

# 步骤2: 在测试集上评估
echo "=== 步骤2: 在测试集上评估 ==="
dataset="sudoku_test"
topk_decoding=True
mkdir -p $exp/$dataset

CUDA_VISIBLE_DEVICES=0 \
python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_predict \
    --cutoff_len 164 \
    --dataset $dataset \
    --finetuning_type full \
    --diffusion_steps 20 \
    --output_dir $exp/${dataset} \
    --checkpoint_dir $exp  \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding $topk_decoding \
    --per_device_eval_batch_size 16 \
    --max_samples 100 \
    > $exp/${dataset}/eval-TopK$topk_decoding.log 2>&1

echo "评估完成！"
echo "训练日志: $exp/train.log"
echo "评估日志: $exp/${dataset}/eval-TopK$topk_decoding.log"
echo "预测结果: $exp/${dataset}/generated_predictions.jsonl"

# 步骤3: 查看准确率
echo "=== 步骤3: 查看准确率 ==="
if [ -f "$exp/${dataset}/generated_predictions.jsonl" ]; then
    echo "预测文件示例:"
    head -5 "$exp/${dataset}/generated_predictions.jsonl"
fi

if [ -f "$exp/${dataset}/predict_results.json" ]; then
    echo "评估结果:"
    cat "$exp/${dataset}/predict_results.json"
fi

echo "=== 测试 pipeline 完成 ==="