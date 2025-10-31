#!/bin/bash

# AMD集群多GPU训练修复脚本
# 解决在AMD集群上使用CPU的问题，启用多卡GPU运行

echo "=== AMD多GPU训练脚本 ==="
echo "检测AMD GPU环境..."

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi

# 设置AMD ROCm环境变量
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_SYSTEM_DB_PATH=""
export HSA_USERPTR_FOR_PAGED_MEM=0

# 检测AMD GPU设备
echo "检测可用的AMD GPU设备..."
if command -v rocm-smi &> /dev/null; then
    echo "找到rocm-smi，检测AMD GPU:"
    rocm-smi --showid
    GPU_COUNT=$(rocm-smi --showid | grep -c "GPU")
    echo "检测到 $GPU_COUNT 个AMD GPU设备"
elif command -v hip-smi &> /dev/null; then
    echo "找到hip-smi，检测AMD GPU:"
    hip-smi
    GPU_COUNT=$(hip-smi | grep -c "GPU")
    echo "检测到 $GPU_COUNT 个AMD GPU设备"
else
    echo "未找到ROCm工具，将使用CPU回退模式"
    GPU_COUNT=0
fi

# 配置参数
model_name=PatchTST_MoE_cluster
GPU_DEVICES="0,1,2,3,4,5,6,7"
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar

echo "=== 训练配置 ==="
echo "模型: $model_name"
echo "GPU设备: $GPU_DEVICES"
echo "数据路径: $root_path_name$data_path_name"

# 根据GPU检测结果设置运行参数
if [ $GPU_COUNT -gt 0 ]; then
    echo "使用AMD GPU多卡训练模式"
    USE_GPU_FLAG="False"  # 使用False避免CUDA检测，但启用多GPU
    USE_MULTI_GPU_FLAG="--use_multi_gpu"
    BATCH_SIZE=32
    DEVICES_ARG="--devices $GPU_DEVICES"
else
    echo "使用CPU训练模式"
    USE_GPU_FLAG="False"
    USE_MULTI_GPU_FLAG=""
    BATCH_SIZE=16
    DEVICES_ARG="--gpu 0"
fi

# 训练循环
echo "=== 开始训练循环 ==="

for seq_len in 96
do
for pred_len in 96 192 336 720
do
for random_seed in 2023
do
for learning_rate in 0.0005
do
for T_num_expert in 16
do
for T_top_k in 1
do
for F_num_expert in 16
do
for F_top_k in 1
do
    echo "=== 训练配置: seq_len=$seq_len, pred_len=$pred_len, lr=$learning_rate ==="
    echo "使用GPU: $USE_GPU_FLAG, 多GPU: $USE_MULTI_GPU_FLAG, 批次大小: $BATCH_SIZE"

    # 设置AMD环境并运行训练
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_SYSTEM_DB_PATH="" \
    HIP_VISIBLE_DEVICES="$GPU_DEVICES" \
    HSA_USERPTR_FOR_PAGED_MEM=0 \
    python -u ../run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --target 0 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --c_out 137 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 64 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --T_num_expert $T_num_expert \
      --T_top_k $T_top_k \
      --F_num_expert $F_num_expert \
      --F_top_k $F_top_k \
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs 100 \
      $DEVICES_ARG \
      $USE_MULTI_GPU_FLAG \
      --use_gpu $USE_GPU_FLAG \
      --itr 1 \
      --batch_size $BATCH_SIZE \
      --learning_rate $learning_rate | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log

    echo "=== 完成: seq_len=$seq_len, pred_len=$pred_len ==="
    echo ""

done
done
done
done
done
done
done
done

echo "=== 所有训练完成 ==="
echo "日志文件保存在: logs/LongForecasting/solar/"