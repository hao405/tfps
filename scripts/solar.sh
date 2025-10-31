if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/solar" ]; then
    mkdir ./logs/LongForecasting/solar
fi


# seq_len=96
model_name=PatchTST_MoE_cluster

GPU=0,1,2,3,4,5,6,7
root_path_name=../dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=solar
data_name=solar

# random_seed=2023


for seq_len in 96
do
for pred_len in 96
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
    MIOPEN_DISABLE_CACHE=1 \
    MIOPEN_SYSTEM_DB_PATH="" \
    HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
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
      --devices 0,1,2,3,4,5,6,7 \
      --use_multi_gpu \
      --use_gpu True \
      --gpu 5\
      --itr 1 --batch_size 32 --learning_rate $learning_rate | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log
done
done
done
done
done
done
done
done


for seq_len in 96
do
for pred_len in 192
do
for random_seed in 2023
do
for learning_rate in 0.0005
do
for T_num_expert in 4
do
for T_top_k in 1
do
for F_num_expert in 4
do
for F_top_k in 1
do

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
      --devices 0,1,2,3,4,5,6,7 \
      --use_multi_gpu \
      --use_gpu True \
      --gpu 5\
      --itr 1 --batch_size 32 --learning_rate $learning_rate | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log
done
done
done
done
done
done
done
done


for seq_len in 96
do
for pred_len in 336
do
for random_seed in 2023
do
for learning_rate in 0.005
do
for T_num_expert in 4
do
for T_top_k in 1
do
for F_num_expert in 4
do
for F_top_k in 1
do


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
      --devices 0,1,2,3,4,5,6,7 \
      --use_multi_gpu \
      --use_gpu True \
      --gpu 5\
      --itr 1 --batch_size 32 --learning_rate $learning_rate | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log
done
done
done
done
done
done
done
done



for seq_len in 96
do
for pred_len in 720
do
for random_seed in 2023
do
for learning_rate in 0.0005
do
for T_num_expert in 8
do
for T_top_k in 1
do
for F_num_expert in 8
do
for F_top_k in 1
do

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
      --devices 0,1,2,3,4,5,6,7 \
      --use_multi_gpu \
      --use_gpu True \
      --gpu 5\
      --itr 1 --batch_size 32 --learning_rate $learning_rate | tee logs/LongForecasting/solar/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${T_num_expert}_${T_top_k}_${F_num_expert}_${F_top_k}_${learning_rate}_0.1.log
done
done
done
done
done
done
done
done