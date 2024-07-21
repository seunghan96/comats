#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
ep=15
model_name=S_Mamba_reg4_wo_conv_wo_TD1

lamb_values=(0 0.00001 0.0001 0.001)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.001 \
    --itr 1
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.001 \
    --itr 1

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.002 \
    --itr 1

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.0008\
    --itr 1
done      