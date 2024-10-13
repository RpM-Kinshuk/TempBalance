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


########################################################################################################
############### Code below handles slicing of Conv2d layers, courtesy of WeightWathcer #################
########################################################################################################

import logging
logger = logging.getLogger(__name__)
UNKNOWN = 'unknown'

class CHANNELS():
    UNKNOWN = UNKNOWN
    FIRST = 'first'
    LAST = 'last'

def channel_str(channel):
        if channel == CHANNELS.FIRST:
            return "FIRST"
        elif channel == CHANNELS.LAST:
            return "LAST"
        else:
            return "UNKNOWN"

def conv2D_Wmats(Wtensor, channels=CHANNELS.LAST):
        """Extract W slices from a 4 layer_id conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  
        Return ij (N x M) matrices, with receptive field size (rf) and channels flag (first or last)"""
        
        logger.debug("conv2D_Wmats")
        
        # TODO:  detect or use CHANNELS
        # if channels specified ...
    
        Wmats = []
        s = Wtensor.shape
        N, M, imax, jmax = s[0], s[1], s[2], s[3]
        
        if N + M >= imax + jmax:
            detected_channels= CHANNELS.LAST
        else:
            detected_channels= CHANNELS.FIRST
            

        if channels == CHANNELS.UNKNOWN :
            logger.debug(f"channels UNKNOWN, detected {channel_str(detected_channels)}")
            channels= detected_channels

        if detected_channels == channels:
            if channels == CHANNELS.LAST:
                logger.debug("channels Last tensor shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[:, :, i, j]
                        if W.shape[0] < W.shape[1]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                        
            else: #channels == CHANNELS.FIRST  # i, j, M, N
                M, N, imax, jmax = imax, jmax, N, M
                # check this       
                logger.debug("channels First shape: {}x{} (NxM), {}x{} (i,j)".format(N, M, imax, jmax))                
                for i in range(imax):
                    for j in range(jmax):
                        W = Wtensor[i, j, :, :]
                        if W.shape[1] < W.shape[0]:
                            N, M = M, N
                            W = W.T
                        Wmats.append(W)
                            
        elif detected_channels != channels:
            logger.warning("warning, expected channels {},  detected channels {}".format(channel_str(channels),channel_str(detected_channels)))
            # flip how we extract the Wmats
            # reverse of above extraction
            if detected_channels == CHANNELS.LAST:
                logger.debug("Flipping LAST to FIRST Channel, {}x{} ()x{}".format(N, M, imax, jmax))   
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[i, j,:,:]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
                        
            else: #detected_channels == CHANNELS.FIRST:
                N, M, imax, jmax = imax, jmax, N, M   
                logger.debug("Flipping FIRST to LAST Channel, {}x{} ()x{}".format(N, M, imax, jmax))                
                # check this       
                for i in range(N):
                    for j in range(M):
                        W = Wtensor[:, :, i, j]
                        if imax < jmax:
                            W = W.T
                        Wmats.append(W)
            # final flip            
            N, M, imax, jmax = imax, jmax, N, M   
           
                
        rf = imax * jmax  # receptive field size             
        logger.debug("get_conv2D_Wmats N={} M={} rf= {} channels= {}".format(N, M, rf, channels))
    
        return Wmats, N, M, rf