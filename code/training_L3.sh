#!/bin/bash

# Parameters
N_arr=(2000)
alpha_P_arr=(0.01 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81 0.91 1.01 1.11 1.21 1.31 1.41 1.51 1.61 1.71 1.81 1.91 2.01 2.11)
alpha_D_arr=(0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 0.21 0.23 0.25)
l_arr=(0.1)  # Lambda values
data_PATH="/home/benedetti/pseudolikelihood/savings_overlaps_L3/"  # Example data path

# Maximum number of processes to run simultaneously
MAX_PROCESSES=15  # Set your desired maximum number of concurrent processes here

# Function to wait if too many processes are running
wait_for_free_slot() {
  while [ "$(jobs | wc -l)" -ge "$MAX_PROCESSES" ]; do
    sleep 1
  done
}

# Loop over parameters
for l in "${l_arr[@]}"; do
  for alpha_P in "${alpha_P_arr[@]}"; do
    for alpha_D in "${alpha_D_arr[@]}"; do
      for N in "${N_arr[@]}"; do
        # Build the command to run the python script with parameters
        cmd="python3 training.py --N $N --alpha_P $alpha_P --alpha_D $alpha_D --l $l --device cpu --data_PATH $data_PATH"

        # Wait for a free slot to run the process
        wait_for_free_slot

        # Run the process in the background
        echo "Running: $cmd"
        $cmd &
      done
    done
  done
done

# Wait for all processes to finish
wait

echo "All processes completed!"
