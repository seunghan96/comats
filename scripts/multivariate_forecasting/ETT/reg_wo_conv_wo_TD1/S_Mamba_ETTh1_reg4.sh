#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
ep=25
model_name=S_Mamba_reg4_wo_conv_wo_TD1

# List of lambda values to iterate over
lamb_values=(0 0.001 0.01 0.1)
#lamb_values=(0.001)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2\
    --d_ff 256 \
    --itr 1 \
    --train_epochs $ep\
    --lamb $lamb\
    --learning_rate 0.00007
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_192 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2 \
    --d_ff 256 \
    --itr 1 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.00007

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_336 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2 \
    --d_ff 256 \
    --itr 1 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.00005
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_720 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_state 2 \
    --d_ff 256 \
    --itr 1 \
    --lamb $lamb\
    --train_epochs $ep\
    --learning_rate 0.00005
  
done    