#!/bin/bash

# AMD Multi-GPU Solar Training Script with Enhanced Debugging
# This script is optimized for AMD GPU parallel training

echo "=== AMD Multi-GPU Solar Training Script ==="
echo "Starting solar training with AMD GPUs"

# Create necessary directories
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi

# Set AMD GPU environment variables for optimal performance
export GPU=0,1,2,3,4,5,6,7
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HIP_VISIBLE_DEVICES=$GPU
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_PATH=/opt/rocm
export HCC_AMDGPU_TARGET=gfx900,gfx906,gfx908,gfx90a,gfx1030

# Debug information
echo "=== GPU Information ==="
echo "Available GPUs: $GPU"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"

# Check if ROCm tools are available
if command -v rocm-smi > /dev/null 2>&1; then
    echo "ROCm GPU Status:"
    rocm-smi --showid --showmeminfo vram --showtemp
else
    echo "Warning: rocm-smi not found, cannot display GPU status"
fi

if command -v hipconfig > /dev/null 2>&1; then
    echo "HIP Configuration:"
    hipconfig --platform
else
    echo "Warning: hipconfig not found"
fi

# Model configuration
model_name=PatchTST_MoE_cluster
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar

# Recommended training config for Solar + PatchTST_MoE_cluster
BATCH_SIZE=32
D_MODEL=16
N_HEADS=8
E_LAYERS=3
D_FF=64
PATCH_LEN=16
STRIDE=8
T_NUM_EXPERT=16
T_TOP_K=1
F_NUM_EXPERT=16
F_TOP_K=1
LR=0.0005
TRAIN_EPOCHS=100
DROPOUT=0.1
FC_DROPOUT=0.1

echo "=== Training Configuration ==="
echo "Model: $model_name"
echo "Batch Size: $BATCH_SIZE"
echo "GPUs: $GPU"
echo "Sequence Length: 96"
echo "Prediction Lengths: 96, 192, 336, 720"

# Training loop with error handling
for seq_len in 96
do
for pred_len in 96 192 336 720
do
for random_seed in 2023
do
for learning_rate in ${LR}
do
for T_num_expert in ${T_NUM_EXPERT}
do
for T_top_k in ${T_TOP_K}
do
for F_num_expert in ${F_NUM_EXPERT}
do
for F_top_k in ${F_TOP_K}
do
    echo "=== Starting Training ==="
    echo "Sequence Length: $seq_len"
    echo "Prediction Length: $pred_len"
    echo "Learning Rate: $learning_rate"
    echo "T_num_expert: $T_num_expert, T_top_k: $T_top_k"
    echo "F_num_expert: $F_num_expert, F_top_k: $F_top_k"

    # Set log file name
    LOG_FILE="logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log"

    echo "Log file: $LOG_FILE"

    # Run training with environment variables and error handling
    if MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_SYSTEM_DB_PATH="" \
    HIP_VISIBLE_DEVICES="$GPU" \
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
      --e_layers ${E_LAYERS} \
      --n_heads ${N_HEADS} \
      --d_model ${D_MODEL} \
      --d_ff ${D_FF} \
      --dropout ${DROPOUT} \
      --fc_dropout ${FC_DROPOUT} \
      --head_dropout 0 \
      --patch_len ${PATCH_LEN} \
      --stride ${STRIDE} \
      --T_num_expert $T_num_expert \
      --T_top_k $T_top_k \
      --F_num_expert $F_num_expert \
      --F_top_k $F_top_k \
      --beta 0.1 \
      --des 'Exp' \
      --train_epochs ${TRAIN_EPOCHS} \
      --devices 0,1,2,3,4,5,6,7 \
      --use_multi_gpu \
      --use_gpu True \
      --gpu 0 \
      --itr 1 --batch_size ${BATCH_SIZE} --learning_rate ${learning_rate}  2>&1 | tee "$LOG_FILE"; then
        echo "✓ Training completed successfully for pred_len=$pred_len"
    else
        echo "✗ Training failed for pred_len=$pred_len"
        echo "Check log file: $LOG_FILE"
        # Continue with next experiment instead of stopping
        continue
    fi

    echo "=== Finished Training for pred_len=$pred_len ==="
    echo ""

done
done
done
done
done
done
done
done

echo "=== All experiments completed ==="
echo "Check logs in: logs/LongForecasting/solar/"