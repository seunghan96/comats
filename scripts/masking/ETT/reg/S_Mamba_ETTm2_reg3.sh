#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
ep=25
model_name=S_Mamba_reg3

# List of lambda values to iterate over
lamb_values=(0)
#lamb_values=(0.001)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"

  # d state 2
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 256 \
    --d_state 2 \
    --learning_rate 0.00005 \
    --lamb $lamb\
    --itr 1

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_192 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_state 2 \
    --learning_rate 0.00005 \
    --lamb $lamb\
    --d_ff 128 \
    --itr 1

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_336 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_state 2 \
    --learning_rate 0.00003 \
    --lamb $lamb\
    --d_ff 128 \
    --itr 1

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_720 \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_state 2 \
    --learning_rate 0.00005 \
    --lamb $lamb\
    --d_ff 128 \
    --itr 1
done    