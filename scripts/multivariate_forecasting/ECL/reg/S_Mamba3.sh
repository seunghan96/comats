#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ep=25
model_name=S_Mamba_reg3

# List of lambda values to iterate over
lamb_values=(0 0.00001)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --d_state 16 \
    --train_epochs 5 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --lamb $lamb\
    --itr 1
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --train_epochs 5 \
    --learning_rate 0.0005 \
    --lamb $lamb\
    --itr 1
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --train_epochs 5 \
    --learning_rate 0.0005 \
    --lamb $lamb\
    --itr 1
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --train_epochs 5 \
    --batch_size 16 \
    --lamb $lamb\
    --learning_rate 0.0005 \
    --itr 1
  
done    



