#!/bin/bash
# 修复的train-mdm.sh脚本
# 修复了数据集参数和DataParallel检查问题
# 保持原始参数：model_config_tiny, bs1024, 8 GPUs

set -e

# 禁用wandb
export WANDB_DISABLED=true

# 创建输出目录
exp=output/sudoku/mdm-alpha0.25-gamma1-bs1536-lr1e-3-ep300-T20-`date "+%Y%m%d-%H%M%S"`
mkdir -p $exp

echo "输出目录: $exp"
echo "使用model_config_tiny模型"
echo ""

# 第一步：修复trainer.py中的DataParallel检查（如果还没修复）
echo "检查并修复trainer.py中的DataParallel检查..."
TRAINER_FILE="src/llmtuner/tuner/mdm/trainer.py"
if [ -f "$TRAINER_FILE" ] && grep -q "if isinstance(model, DDP):" "$TRAINER_FILE"; then
    echo "修复DataParallel检查..."
    sed -i "s/if isinstance(model, DDP):/if hasattr(model, 'module'):/" "$TRAINER_FILE"
    echo "✅ DataParallel检查已修复"
else
    echo "✅ trainer.py已正确配置"
fi

echo ""

# 第二步：运行训练
echo "开始训练（8个GPU，每个批次200，总批次1600）..."
echo "训练日志: $exp/train.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 --num_processes 8 --main_process_port 20099 \
src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset sudoku_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir $exp \
    --overwrite_cache \
    --per_device_train_batch_size 200 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --plot_loss \
    --run_name sudoku_mdm_tiny \
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
echo ""

# 第三步：评估
echo "开始评估..."

for dataset in sudoku_test
do
    topk_decoding=True
    mkdir -p $exp/$dataset
    echo "评估数据集: $dataset"

    CUDA_VISIBLE_DEVICES=1  \
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
        > $exp/${dataset}/eval-TopK$topk_decoding.log 2>&1

    echo "评估完成: $dataset"
    echo "评估日志: $exp/${dataset}/eval-TopK$topk_decoding.log"
done

echo ""
echo "=========================================="
echo "实验完成!"
echo "输出目录: $exp"
echo ""
echo "检查训练结果:"
echo "  tail -20 $exp/train.log"
echo "检查评估结果:"
echo "  tail -20 $exp/sudoku_test/eval-TopKTrue.log"
echo "=========================================="