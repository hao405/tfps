#!/bin/bash

# AMD集群专用多GPU训练脚本
# 专门解决solar.sh在AMD集群上使用CPU的问题

echo "========================================="
echo "AMD集群多GPU训练脚本 - Solar数据集"
echo "========================================="

# 基础目录设置
cd "$(dirname "$0")" || exit 1

# 创建日志目录
mkdir -p logs/LongForecasting/solar

# 检测AMD GPU环境
echo "正在检测AMD GPU环境..."

# 方法1: 使用rocm-smi检测AMD GPU
if command -v rocm-smi &> /dev/null; then
    echo "✓ 发现rocm-smi工具"
    echo "AMD GPU设备列表:"
    rocm-smi --showid
    AVAILABLE_GPUS=$(rocm-smi --showid 2>/dev/null | grep "GPU" | wc -l)
    echo "检测到 $AVAILABLE_GPUS 个AMD GPU设备"

# 方法2: 使用hip-smi检测
elif command -v hip-smi &> /dev/null; then
    echo "✓ 发现hip-smi工具"
    hip-smi
    AVAILABLE_GPUS=$(hip-smi 2>/dev/null | grep -c "GPU")
    echo "检测到 $AVAILABLE_GPUS 个AMD GPU设备"

# 方法3: 检查HIP环境
elif [ -n "$HIP_VISIBLE_DEVICES" ]; then
    echo "✓ 发现HIP_VISIBLE_DEVICES环境变量"
    AVAILABLE_GPUS=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)
    echo "配置显示 $AVAILABLE_GPUS 个GPU设备"

# 方法4: 检查设备文件
elif [ -d "/dev/kfd" ] || [ -d "/dev/dri" ]; then
    echo "✓ 发现AMD GPU设备文件"
    AVAILABLE_GPUS=8  # 假设标准8卡配置
    echo "默认使用 $AVAILABLE_GPUS 个GPU设备"

else
    echo "⚠ 未检测到AMD GPU，将使用CPU模式"
    AVAILABLE_GPUS=0
fi

# 根据检测结果配置GPU
if [ $AVAILABLE_GPUS -gt 0 ]; then
    # 构建GPU设备列表 (0,1,2,3,4,5,6,7)
    GPU_LIST=""
    for ((i=0; i<AVAILABLE_GPUS; i++)); do
        if [ $i -eq 0 ]; then
            GPU_LIST="$i"
        else
            GPU_LIST="$GPU_LIST,$i"
        fi
    done
    echo "使用GPU设备: $GPU_LIST"
else
    GPU_LIST="0"
    echo "使用CPU设备"
fi

# 设置AMD ROCm环境变量
echo "设置ROCm环境变量..."
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_VISIBLE_DEVICES="$GPU_LIST"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_SYSTEM_DB_PATH=""
export HSA_USERPTR_FOR_PAGED_MEM=0
export ROCR_VISIBLE_DEVICES="$GPU_LIST"

# 训练参数配置
model_name=PatchTST_MoE_cluster
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar

echo "========================================="
echo "训练参数配置:"
echo "模型: $model_name"
echo "数据: $root_path_name$data_path_name"
echo "GPU设备: $GPU_LIST ($AVAILABLE_GPUS 设备)"
echo "========================================="

# 根据硬件配置优化参数
if [ $AVAILABLE_GPUS -ge 8 ]; then
    # 8卡配置
    BATCH_SIZE=32
    USE_GPU="False"  # 关键：使用False避免CUDA检测，但启用多GPU
    USE_MULTI_GPU="--use_multi_gpu"
    DEVICES_ARG="--devices $GPU_LIST"
    echo "使用8卡AMD GPU优化配置"

elif [ $AVAILABLE_GPUS -ge 4 ]; then
    # 4卡配置
    BATCH_SIZE=64
    USE_GPU="False"
    USE_MULTI_GPU="--use_multi_gpu"
    DEVICES_ARG="--devices $GPU_LIST"
    echo "使用4卡AMD GPU优化配置"

elif [ $AVAILABLE_GPUS -ge 2 ]; then
    # 2卡配置
    BATCH_SIZE=128
    USE_GPU="False"
    USE_MULTI_GPU="--use_multi_gpu"
    DEVICES_ARG="--devices $GPU_LIST"
    echo "使用2卡AMD GPU优化配置"

else
    # CPU配置
    BATCH_SIZE=16
    USE_GPU="False"
    USE_MULTI_GPU=""
    DEVICES_ARG="--gpu 0"
    echo "使用CPU训练模式"
fi

# 训练循环
echo "开始训练循环..."
echo "========================================="

# 主要训练配置
seq_len=96
pred_lens=(96 192 336 720)
learning_rate=0.0005
T_num_expert=16
T_top_k=1
F_num_expert=16
F_top_k=1
random_seed=2023

for pred_len in "${pred_lens[@]}"
do
    echo ""
    echo ">>> 训练配置: seq_len=$seq_len, pred_len=$pred_len <<<"
    echo "GPU使用: $USE_GPU, 多GPU: $USE_MULTI_GPU, 批次大小: $BATCH_SIZE"
    echo "设备参数: $DEVICES_ARG"
    echo ""

    # 构建日志文件名
    log_file="logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log"

    echo "日志文件: $log_file"
    echo "开始训练..."

    # 执行训练命令
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_SYSTEM_DB_PATH="" \
    HIP_VISIBLE_DEVICES="$GPU_LIST" \
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
      $USE_MULTI_GPU \
      --use_gpu $USE_GPU \
      --itr 1 \
      --batch_size $BATCH_SIZE \
      --learning_rate $learning_rate 2>&1 | tee "$log_file"

    # 检查训练是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ 训练成功完成: seq_len=$seq_len, pred_len=$pred_len"
    else
        echo "✗ 训练失败: seq_len=$seq_len, pred_len=$pred_len"
        echo "检查日志文件: $log_file"
    fi

    echo "========================================="
    echo ""

done

echo "========================================="
echo "所有训练任务完成！"
echo "日志文件保存在: logs/LongForecasting/solar/"
echo "========================================="

# 显示训练总结
echo "训练总结:"
ls -la logs/LongForecasting/solar/*.log | wc -l | xargs echo "完成的训练任务数:"
echo "最近的日志文件:"
ls -lt logs/LongForecasting/solar/*.log | head -3

# 可选：清理环境变量
echo "清理环境变量..."
unset HIP_VISIBLE_DEVICES
unset MIOPEN_DISABLE_CACHE
unset MIOPEN_SYSTEM_DB_PATH
unset HSA_USERPTR_FOR_PAGED_MEM

echo "脚本执行完成！"