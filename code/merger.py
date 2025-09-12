import os
import numpy as np

# Define parameters
N = 1000
alpha_P_arr = [0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91, 2.01, 2.11]
alpha_D_arr = [0.01, 0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 1.81, 1.91, 2.01, 2.11]
l = 0.1

# Define file path pattern
data_PATH = "savings_overlaps/"  # Update this to the correct path if different

# Initialize result arrays
vali_loss_array = np.zeros((len(alpha_P_arr), len(alpha_D_arr)))
vali_loss_f_array = np.zeros((len(alpha_P_arr), len(alpha_D_arr)))

# Iterate over parameter combinations
for i, alpha_P in enumerate(alpha_P_arr):
    for j, alpha_D in enumerate(alpha_D_arr):
        # Construct the file name
        model_name = f"GD_capacity_N_{N}_P_{int(alpha_P*1000)}_D{int(alpha_D*1000)}_l_{l}_overlaps.npy"
        file_path = os.path.join(data_PATH, model_name)
        
        try:
            # Load the file
            data = np.load(file_path)
            vali_loss_array[i, j] = data[0]  # vali_loss
            vali_loss_f_array[i, j] = data[1]  # vali_loss_f
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping...")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Save the result arrays if needed
np.save(os.path.join(data_PATH, "vali_loss_array_binary_lr1000_ep800_ov1_2.npy"), vali_loss_array)
np.save(os.path.join(data_PATH, "vali_loss_f_array_binary_lr1000_ep800_ov1_2.npy"), vali_loss_f_array)

# Print shapes and a preview
print("vali_loss_array shape:", vali_loss_array.shape)
print("vali_loss_f_array shape:", vali_loss_f_array.shape)
print("Sample vali_loss_array:", vali_loss_array)
print("Sample vali_loss_f_array:", vali_loss_f_array)
