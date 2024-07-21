#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
ep=15
model_name=S_Mamba_reg3_wo_conv_wo_TD1

# List of lambda values to iterate over
lamb_values=(0.0001 0.001 0.01)

for lamb in "${lamb_values[@]}"
do
  echo "Running with lamb=$lamb"
  # d_state = 32

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16\
    --learning_rate 0.0007 \
    --train_epochs $ep \
    --lamb $lamb\
    --itr 1 \
    --use_norm 0

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_24 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.0007 \
    --train_epochs $ep \
    --lamb $lamb\
    --itr 1 \
    --use_norm 0

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_48 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 4 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.0005 \
    --train_epochs $ep \
    --lamb $lamb\
    --itr 1 \
    --use_norm 0
    --batch_size 16\
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16\
    --learning_rate 0.0005 \
    --train_epochs $ep \
    --lamb $lamb\
    --itr 1 \
    --use_norm 0
done