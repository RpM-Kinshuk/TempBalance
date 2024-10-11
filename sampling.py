import math
import torch

########################################################################################################
# This function applies sliding window sampling matrix method for approximating random matrix ESD (V1) #
########################################################################################################
def matrix_size_dependent_number_of_sampling_ops(matrix, num_row_samples, Q_ratio, step_size=10):
    rows, cols = matrix.shape
    
    num_col_samples = int(num_row_samples * Q_ratio)
    
    # Determine the number of windows
    num_row_windows = max(1, (rows - num_row_samples) // step_size + 1)
    num_col_windows = max(1, (cols - num_col_samples) // step_size + 1)
    
    all_eigs = []
    # print(f"Matrix shape: ({rows}, {cols}) | Number of sliding operations across rows: {num_row_windows} and across columns: {num_col_windows}")
    
    for i in range(0, num_row_windows):
        for j in range(0, num_col_windows):
            # Extract submatrix
            row_start = i * step_size
            col_start = j * step_size
            submatrix = matrix[row_start:row_start + num_row_samples, 
                               col_start:col_start + num_col_samples]
            
            # Compute singular values
            s = torch.linalg.svdvals(submatrix)
            
            # Compute eigenvalues (squared singular values)
            eigs = torch.square(s)
            
            all_eigs.append(eigs)
    
    # Concatenate all eigenvalues
    all_eigs = torch.cat(all_eigs)
    
    # Sort eigenvalues
    all_eigs, _ = torch.sort(all_eigs, descending=False)
    
    # Return eigs as a tensor on the same device
    return all_eigs

########################################################################################################
# This function applies sliding window sampling matrix method for approximating random matrix ESD (V2) #
########################################################################################################
def fixed_number_of_sampling_ops(matrix, num_row_samples, Q_ratio, num_sampling_ops_per_dimension):
    rows, cols = matrix.shape
    
    num_col_samples = int(num_row_samples * Q_ratio)
    
    # Calculate step sizes
    row_step = max(1, math.floor((rows - num_row_samples) / (num_sampling_ops_per_dimension - 1)))
    col_step = max(1, math.floor((cols - num_col_samples) / (num_sampling_ops_per_dimension - 1)))
    
    # Determine the actual number of windows based on calculated step sizes
    num_row_windows = min(num_sampling_ops_per_dimension, max(1, (rows - num_row_samples) // row_step + 1))
    num_col_windows = min(num_sampling_ops_per_dimension, max(1, (cols - num_col_samples) // col_step + 1))
    
    all_eigs = []
    # print(f"Matrix shape: ({rows}, {cols}) | Number of sliding operations across rows: {num_row_windows} and across columns: {num_col_windows}")
    # print(f"Row step size: {row_step}, Column step size: {col_step}")
    
    for i in range(0, num_row_windows):
        for j in range(0, num_col_windows):
            # Extract submatrix
            row_start = min(i * row_step, rows - num_row_samples)
            col_start = min(j * col_step, cols - num_col_samples)
            submatrix = matrix[row_start:row_start+num_row_samples, 
                               col_start:col_start+num_col_samples]
            
            # Compute singular values
            s = torch.linalg.svdvals(submatrix)
            
            # Compute eigenvalues (squared singular values)
            eigs = torch.square(s)
            
            all_eigs.append(eigs)
    
    # Concatenate all eigenvalues
    all_eigs = torch.cat(all_eigs)
    
    # Sort eigenvalues
    all_eigs, _ = torch.sort(all_eigs, descending=False)
    
    # Return eigs as a tensor on the same device
    return all_eigs