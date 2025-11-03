#!/bin/bash

# Solar dataset debugging script with safer hyperparameters

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi

model_name=PatchTST_MoE_cluster

GPU=0,1,2,3,4,5,6,7
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HIP_VISIBLE_DEVICES=$GPU

# Safer training config to avoid NaN loss
BATCH_SIZE=16          # 减小batch size
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
LR=0.000001            # 降低学习率 (从0.00001降到0.000001)
TRAIN_EPOCHS=100
DROPOUT=0.2            # 增加dropout
FC_DROPOUT=0.2

# 测试单个配置
seq_len=96
pred_len=96
random_seed=2023
learning_rate=${LR}
T_num_expert=${T_NUM_EXPERT}
T_top_k=${T_TOP_K}
F_num_expert=${F_NUM_EXPERT}
F_top_k=${F_TOP_K}

echo "=== Starting Solar Training with Debug Configuration ==="
echo "Learning Rate: ${learning_rate}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Dropout: ${DROPOUT}"

MIOPEN_DISABLE_CACHE=1 \
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
  --beta 0.01 \
  --alpha 0.1 \
  --gama 0.1 \
  --des 'Debug' \
  --train_epochs ${TRAIN_EPOCHS} \
  --devices 0,1,2,3,4,5,6,7 \
  --use_multi_gpu \
  --use_gpu True \
  --gpu 0 \
  --itr 1 \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${learning_rate} 2>&1 | tee logs/LongForecasting/solar/${model_name}_debug_${seq_len}_${pred_len}.log

echo "Debug run completed. Check the log file for details."
