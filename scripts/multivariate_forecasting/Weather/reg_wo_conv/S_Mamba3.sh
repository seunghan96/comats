#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
ep=25
model_name=S_Mamba_reg3_wo_conv

# List of lambda values to iterate over
lamb_values=(0 0.0001 0.001 0.01 0.1)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512\
    --learning_rate 0.00005 \
    --train_epochs $ep\
    --d_state 2 \
    --d_ff 512\
    --lamb $lamb\
    --itr 1


  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --learning_rate 0.00005 \
    --train_epochs $ep\
    --d_model 512\
    --d_state 2 \
    --d_ff 512\
    --lamb $lamb\
    --itr 1


  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --learning_rate 0.00005 \
    --train_epochs $ep\
    --d_model 512\
    --d_state 2 \
    --d_ff 512\
    --lamb $lamb\
    --itr 1


  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --learning_rate 0.00005 \
    --train_epochs $ep\
    --d_model 512\
    --d_state 2 \
    --d_ff 512\
    --lamb $lamb\
    --itr 1
done