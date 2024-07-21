#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
ep=25
model_name=S_Mamba_reg

# List of lambda values to iterate over
#lamb_values=(0.01 0.1 0.2 0.5 1.0)
lamb_values=(0.001)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"

  python -u run.py \
    --is_training 0 \
    --multiple_inference 10\
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data ETTh2 \
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
    --learning_rate 0.00004 \
    --lamb $lamb\
    --itr 1

  
done