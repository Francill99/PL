#!/bin/bash

# Parameters
N=1000
alpha_P=1.0
alpha_D=0.05
l=1.0
device="cpu"
epochs=401
learning_rate=1.0
init_overlap=1.0
data_PATH="savings"

LOG_FILE="training_log.txt"

# Clear previous log
: > "$LOG_FILE"

# Run in background with nohup, unbuffered (-u), logs written in real time
nohup python3 -u training.py \
  --N $N \
  --alpha_P $alpha_P \
  --alpha_D $alpha_D \
  --l $l \
  --device $device \
  --epochs $epochs \
  --learning_rate $learning_rate \
  --init_overlap $init_overlap \
  --data_PATH $data_PATH \
  > "$LOG_FILE" 2>&1 &

echo "Training started in background. Logs: $LOG_FILE"
echo "PID: $!"
