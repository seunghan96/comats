#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
ep=15
model_name=S_Mamba_reg4

# List of lambda values to iterate over
lamb_values=(0 0.0001 0.001 0.01)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  # d_state = 32

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0003 \
    --itr 1 \
    --train_epochs $ep \
    --lamb $lamb\
    --use_norm 0

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_24 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0003 \
    --itr 1 \
    --train_epochs $ep \
    --lamb $lamb\
    --use_norm 0

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_48 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0003 \
    --itr 1 \
    --train_epochs $ep \
    --lamb $lamb\
    --use_norm 0

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --learning_rate 0.0003 \
    --train_epochs $ep \
    --lamb $lamb\
    --itr 1 \
    --use_norm 0
done