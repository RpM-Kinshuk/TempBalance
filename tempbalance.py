import torch
import sys
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from operator import itemgetter
from sampling import *


class Tempbalance(object):
    def __init__(self, 
                    net, 
                    EVALS_THRESH=0.00001,
                    bins=100, 
                    conv_norm=0.5,
                    pl_fitting='median',
                    xmin_pos=2,
                    filter_zeros=False,
                    remove_first_layer=True,
                    remove_last_layer=True,
                    eigs_thresh=50,
                    esd_metric_for_tb='alpha',
                    assign_func='tb_linear_map',
                    lr_min_ratio=0.5,
                    lr_max_ratio=1.5,
                    batchnorm=True,
                    batchnorm_type='name',
                    layernorm=False,
                    use_sliding_window=False,
                    num_row_samples=100,  # Required for sliding window
                    Q_ratio=2.0,  # Required for sliding window
                    step_size=10,  # Sliding window step size for variable ops
                    sampling_ops_per_dim=None,  # For fixed number of operations
                    ):
        """init function
        Args:
            net (nn.module):             net to train
            EVALS_THRESH (float, ):      threshold to filter small eigenvalue. Defaults to 0.00001.
            bins (int, int):             ESD bins. Defaults to 100.
            conv_norm (float, ):         conv norm. Defaults to 0.5.
            pl_fitting (str, ):          powerlaw fitting method. Defaults to median, ['median', 'goodness-of-fit', 'fix-finger']
            xmin_pos (int, ):            set the position of minimum eigenvalue in the tail. Defaults to 2.
            filter_zeros (bool, ):       filter small eigenvalues or not. Defaults to False.
            remove_first_layer (bool, ): whether exclude first layer in TB. Defaults to True.
            remove_last_layer (bool, ): whether exclude last layer in TB. Defaults to True.
            esd_metric_for_tb (str, ): metric for TB scheduling. Defaults to 'alpha'.
            assign_func (str, ):         learning rate assignment function. Defaults to 'tb_linear_map'.
            lr_min_ratio (float, ):      learning rate lower bound. Defaults to 0.5.
            lr_max_ratio (float, ):       learning rate upper bound. Defaults to 1.5.
            batchnorm (bool, ):          whether adjust batch norm learning rate using TB. Defaults to True.
            batchnorm_type (str, ):      how to set learning rate for batchnorm layers
            layernorm (bool, ):          whether adjust layer norm learning rate using TB. Defaults to True.
            use_sliding_window (bool, ): whether use sliding window for ESD sampling. Defaults to False.
            num_row_samples (int, ):     number of rows to sample in sliding window. Defaults to 100.
            Q_ratio (float, ):           ratio of sampled columns to rows in sliding window. Defaults to 2.0.
            step_size (int, ):           step size for sliding window in variable ops mode. Defaults to 10.
            sampling_ops_per_dim (int, ): number of sampling operations for fixed ops mode. Defaults to None.
        """
        self.net = net
        self.EVALS_THRESH = EVALS_THRESH
        self.bins = bins
        self.conv_norm = conv_norm
        self.pl_fitting = pl_fitting
        self.xmin_pos = xmin_pos
        self.filter_zeros = filter_zeros
        self.remove_first_layer = remove_first_layer
        self.remove_last_layer = remove_last_layer
        self.eigs_thresh = eigs_thresh
        self.esd_metric_for_tb = esd_metric_for_tb
        self.assign_func = assign_func
        self.lr_min_ratio = lr_min_ratio
        self.lr_max_ratio = lr_max_ratio
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.use_sliding_window = use_sliding_window
        self.num_row_samples = num_row_samples
        self.Q_ratio = Q_ratio
        self.step_size = step_size
        self.sampling_ops_per_dim = sampling_ops_per_dim
        self.bn_to_conv = {}
        self.ln_to_linear = {}
        # print('EVALS_THRESH',  self.EVALS_THRESH, type(self.EVALS_THRESH) )
        # print('bins',  self.bins, type(self.bins) )
        # print('conv_norm',  self.conv_norm, type(self.conv_norm) )
        # print('pl_fitting',  self.pl_fitting, type(self.pl_fitting) )
        # print('xmin_pos',  self.xmin_pos, type(self.xmin_pos) )
        # print('filter_zeros',  self.filter_zeros, type(self.filter_zeros) )
        # print('remove_first_layer',  self.remove_first_layer, type(self.remove_first_layer) )
        # print('remove_last_layer',  self.remove_last_layer, type(self.remove_last_layer) )
        # print('esd_metric_for_tb',  self.esd_metric_for_tb, type(self.esd_metric_for_tb) )
        # print('assign_func',  self.assign_func, type(self.assign_func) )
        # print('lr_min_ratio',  self.lr_min_ratio, type(self.lr_min_ratio) )
        # print('lr_max_ratio',  self.lr_max_ratio, type(self.lr_max_ratio) )
        # print('batchnorm',  self.batchnorm, type(self.batchnorm) )
        
        if batchnorm and batchnorm_type == 'name':
            # let the batch norm layer change lr corresponding to the layer
            # with the same layer name 
            longname_lst = []
            for name, m in self.net.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    longname_lst.append(name)
            for name, module in self.net.named_modules():
                if isinstance(module, nn.BatchNorm2d) \
                        and name.replace('bn', 'conv') in longname_lst:
                    self.bn_to_conv[name] = name.replace('bn', 'conv')
                    
        elif batchnorm and batchnorm_type == 'order':
            # let the batch norm layer change lr corresponding to the 
            # conv layer before current layer
            longname_lst = []
            type_lst = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Conv2d):
                    longname_lst.append(name)
                    type_lst.append('nn.Conv2d')
                if isinstance(module, nn.BatchNorm2d):
                    if type_lst[-1] == 'nn.Conv2d':
                        self.bn_to_conv[name] = longname_lst[-1]
                    longname_lst.append(name)
                    type_lst.append('nn.BatchNorm2d')
        
        if self.layernorm:
            longname_lst = []
            type_lst = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Linear):
                    longname_lst.append(name)
                    type_lst.append('nn.Linear')
                if isinstance(module, nn.LayerNorm):
                    if type_lst[-1] == 'nn.Linear':
                        self.ln_to_linear[name] = longname_lst[-1]
                    longname_lst.append(name)
                    type_lst.append('nn.LayerNorm')
            
        
    def build_optimizer_param_group(self, untuned_lr=0.1, initialize=True):
        """build the parameter group for optimizer

        Args:
            untuned_lr (float, ): global learning rate that is not tuned. Defaults to 0.1.
            initialize (bool, ): if True, build a list of dictionary, if False, build a list of learning rate . Defaults to True.

        Returns:
            _type_: _description_
        """
        metrics = self.net_esd_estimator()
        layer_stats = pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
        
        if self.remove_first_layer:
            layer_stats = layer_stats.drop(labels=0, axis=0)
            # index must be reset otherwise may delete the wrong row 
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        if self.remove_last_layer:
            layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0)
            # index must be reset otherwise may delete the wrong row 
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        
        # remove layers with number of eigs less than a threshold
        layer_stats = layer_stats[layer_stats['eigs_num'] >= self.eigs_thresh]
        layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        
        metric_scores = np.array(layer_stats[self.esd_metric_for_tb])
        scheduled_lr = self.get_layer_temps(assign_func=self.assign_func, 
                                            metric_scores=metric_scores, 
                                            untuned_lr=untuned_lr)
        
        layer_stats['scheduled_lr'] = scheduled_lr
        layer_name_to_tune = list(layer_stats['longname'])
        opt_params_groups = []
        params_to_tune_ids = []
        layer_count = 0
        # these params should be tuned
        for name, module in self.net.named_modules():
            
            # these are the conv layers analyzed by ESD
            if name in layer_name_to_tune:
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
                if initialize:
                    # append a dictionary for initialize optimizer
                    opt_params_groups.append({'params': module.parameters(), 'lr': scheduled_lr})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
            # decide should we tune the batch norm accordingly
            elif self.batchnorm \
                and isinstance(module, nn.BatchNorm2d) \
                    and name in self.bn_to_conv \
                        and self.bn_to_conv[name] in layer_name_to_tune:
                
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == self.bn_to_conv[name]]['scheduled_lr'].item()
                if initialize:
                    # append a dictionary for initialize optimizer
                    opt_params_groups.append({'params': module.parameters(), 'lr': scheduled_lr})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
            
            elif self.layernorm \
                and isinstance(module, nn.LayerNorm) \
                    and name in self.ln_to_linear \
                        and self.ln_to_linear[name] in layer_name_to_tune:
                
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_lr = layer_stats[layer_stats['longname'] == self.ln_to_linear[name]]['scheduled_lr'].item()
                if initialize:
                    opt_params_groups.append({'params': module.parameters(), 'lr': scheduled_lr})
                else:
                    opt_params_groups.append(scheduled_lr)
                layer_count += 1
        
        if initialize:
            # those params are untuned
            untuned_params = \
                filter(lambda p: id(p) not in params_to_tune_ids, self.net.parameters())
            opt_params_groups.append({'params': untuned_params, 'lr': untuned_lr}) 
            return opt_params_groups, layer_count
        else:
            return opt_params_groups, layer_count
    
    def step(self, optimizer, untuned_lr):
        opt_params_groups, layer_count = \
            self.build_optimizer_param_group(untuned_lr=untuned_lr, initialize=False)
        for index, param_group in enumerate(optimizer.param_groups):
            if index <= layer_count - 1:
                param_group['lr'] = opt_params_groups[index]
            else:
                param_group['lr'] = untuned_lr
    
    def tempbal_esd_estimator(
            self,
            verbose=False):
        """evaluate the ESD of the conv nets
        Args:
            verbose: 
        Returns:
            _type_: _description_
        """
        results = {
            'alphahat':[],
            'alpha':[],
            'spectral_norm': [],
            'D': [],
            'longname':[],
            'eigs':[],
            'norm':[],
            'eigs_num':[]
            }
        if verbose:
            print("=================================")
            print(f"pl_fitting: {self.pl_fitting}, xmin_pos: {self.xmin_pos}, conv_norm: {self.conv_norm}, filter_zeros: {self.filter_zeros}")
            print("=================================")
        # iterate through layers
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                matrix = m.weight.data.clone()
                # normalization and tranpose Conv2d
                if isinstance(m, nn.Conv2d):
                    matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(self.conv_norm)
                    matrix = matrix.transpose(1, 2).transpose(0, 1)
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
                # ascending order 
                eigs, _ = torch.sort(eigs, descending=False)
                spectral_norm = eigs[-1].item()
                fnorm = torch.sum(eigs).item()
                
                if self.filter_zeros:
                    nz_eigs = eigs[eigs > self.EVALS_THRESH]
                    N = len(nz_eigs)
                    # somethines N may equal 0, if that happens, we don't filter eigs
                    if N == 0:
                        nz_eigs = eigs
                        N = len(nz_eigs)
                else:
                    nz_eigs = eigs
                    N = len(nz_eigs)

                log_nz_eigs  = torch.log(nz_eigs)

                if self.pl_fitting == 'median':
                    i = int(len(nz_eigs) / self.xmin_pos)    
                    xmin = nz_eigs[i]
                    n = float(N - i)
                    seq = torch.arange(n).cuda()
                    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    final_D = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                            ))
                else:
                    alphas = torch.zeros(N-1)
                    Ds     = torch.ones(N-1)
                    if self.pl_fitting == 'fix-finger':
                        hist_nz_eigs = torch.log10(nz_eigs)
                        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                        counts = torch.histc(hist_nz_eigs, self.bins, min=min_e, max=max_e)
                        boundaries = torch.linspace(min_e, max_e, self.bins + 1)
                        h = counts, boundaries
                        ih = torch.argmax(h[0])  # 
                        xmin2 = 10 ** h[1][ih]
                        xmin_min = torch.log10(0.95 * xmin2)
                        xmin_max = 1.5 * xmin2
                    
                    for i, xmin in enumerate(nz_eigs[:-1]):
                        if self.pl_fitting == 'fix-finger':
                            if xmin < xmin_min:
                                continue
                            if xmin > xmin_max:
                                break

                        n = float(N - i)
                        seq = torch.arange(n).cuda()
                        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                        alphas[i] = alpha
                        if alpha > 1:
                            Ds[i] = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                            ))

                    min_D_index = torch.argmin(Ds)
                    final_alpha = alphas[min_D_index]
                    final_D = Ds[min_D_index]
                
                final_alpha = final_alpha.item()
                final_D = final_D.item()
                final_alphahat=final_alpha*math.log10(spectral_norm)

                results['spectral_norm'].append(spectral_norm)
                results['alphahat'].append(final_alphahat)
                results['norm'].append(fnorm)
                results['alpha'].append(final_alpha)
                results['D'].append(final_D)
                results['longname'].append(name)
                results['eigs'].append(eigs.detach().cpu().numpy())
                results['eigs_num'].append(len(eigs))
        
        return results
            
    def net_esd_estimator(
        self,
        fix_fingers='xmin_mid',
        verbose=False
    ):
        """Estimator for Empirical Spectral Density (ESD) and Alpha parameter.
    
        Args:
            net (nn.Module): Model to evaluate.
            EVALS_THRESH (float, optional): Threshold to filter near-zero eigenvalues. Defaults to 0.00001.
            bins (int, optional): Number of bins for histogram. Defaults to 100.
            fix_fingers (str, optional): 'xmin_peak' or 'xmin_mid'. Method to select xmin.
            xmin_pos (int, optional): Position in eigenvalue spectrum to choose xmin. Defaults to 2.
            conv_norm (float, optional): Normalization for convolutional layers. Defaults to 0.5.
            filter_zeros (bool, optional): Whether to filter zero eigenvalues. Defaults to False.
            use_sliding_window (bool, optional): Whether to use sliding window sampling. Defaults to False.
            num_row_samples (int, optional): Number of rows to sample in sliding window.
            Q_ratio (float, optional): Ratio of sampled columns to rows in sliding window.
            step_size (int, optional): Step size for sliding window in variable ops mode.
            sampling_ops_per_dim (int, optional): Number of sampling operations for fixed ops mode.
    
        Returns:
            dict: Results containing spectral norm, alpha values, and other metrics.
        """
        
        net = self.net
        EVALS_THRESH = self.EVALS_THRESH
        conv_norm = self.conv_norm
        bins = self.bins
        use_sliding_window = self.use_sliding_window
        num_row_samples = self.num_row_samples
        Q_ratio = self.Q_ratio
        step_size = self.step_size
        sampling_ops_per_dim = self.sampling_ops_per_dim

        if not use_sliding_window:
            results = self.tempbal_esd_estimator(verbose=verbose)
            return results

        results = {
            'alpha': [],
            'spectral_norm': [],
            'D': [],
            'longname': [],
            'eigs': [],
            'norm': [],
            'alphahat': [],
            'eigs_num': []
        }
        print("======================================")
        print(f"fix_fingers: {fix_fingers}, xmin_pos: {self.xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {self.filter_zeros}")
        print(f"use_sliding_window: {use_sliding_window}, num_row_samples: {num_row_samples}, Q_ratio: {Q_ratio}, step_size: {step_size}, sampling_ops_per_dim: {sampling_ops_per_dim}")
        print("======================================")
    
        device = next(net.parameters()).device  # type: ignore
    
        for name, m in net.named_modules():  # type: ignore
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                matrix = m.weight.data.clone().to(device)
                matrix = matrix.float()

                # Sliding window option for sampling ESD
                if use_sliding_window:
                    if isinstance(m, nn.Conv2d):
                        # Flatten the Conv2d weight tensor to 2D
                        matrix = matrix.view(matrix.size(0), -1) * math.sqrt(conv_norm)
                    if sampling_ops_per_dim is not None:
                        print(f"Shape of matrix for {name}: {matrix.shape}")
                        if matrix.dim() != 2:
                            raise ValueError(f"Expected a 2D matrix, got {matrix.dim()} dimensions for {name}.")
                        eigs = fixed_number_of_sampling_ops(
                            matrix, 
                            num_row_samples=num_row_samples, 
                            Q_ratio=Q_ratio, 
                            num_sampling_ops_per_dimension=sampling_ops_per_dim, 
                        )
                    else:
                        eigs = matrix_size_dependent_number_of_sampling_ops(
                            matrix, 
                            num_row_samples=num_row_samples, 
                            Q_ratio=Q_ratio, 
                            step_size=step_size,
                        )
                else:
                    if isinstance(m, nn.Conv2d):
                        matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                        matrix = matrix.transpose(1, 2).transpose(0, 1)
                    matrix = matrix.float()
                    # Regular full matrix ESD computation
                    eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
                    eigs = torch.sort(eigs).values
                
                if not isinstance(eigs, torch.Tensor):
                    eigs = torch.tensor(eigs, device=device)
                spectral_norm = eigs[-1].item()
                fnorm = torch.sum(eigs).item()
    
                # Filtering based on EVALS_THRESH
                nz_eigs = eigs[eigs > EVALS_THRESH] if self.filter_zeros else eigs
                if len(nz_eigs) == 0:
                    nz_eigs = eigs
                N = len(nz_eigs)
                log_nz_eigs = torch.log(nz_eigs)
    
                # Proceed with alpha and D calculations
                if fix_fingers == 'xmin_mid':
                    i = N // self.xmin_pos
                    xmin = nz_eigs[i]
                    n = float(N - i)
                    seq = torch.arange(n, device=device)
                    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
                else:
                    alphas = torch.zeros(N-1, device=device)
                    Ds = torch.ones(N-1, device=device)
                    if fix_fingers == 'xmin_peak':
                        hist_nz_eigs = torch.log10(nz_eigs)
                        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                        counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e) # type: ignore
                        boundaries = torch.linspace(min_e, max_e, bins + 1) # type: ignore
                        ih = torch.argmax(counts)
                        xmin2 = 10 ** boundaries[ih]
                        xmin_min = torch.log10(0.95 * xmin2)
                        xmin_max = 1.5 * xmin2
                    
                    for i, xmin in enumerate(nz_eigs[:-1]):
                        if fix_fingers == 'xmin_peak':
                            if xmin < xmin_min:
                                continue
                            if xmin > xmin_max:
                                break
                        n = float(N - i)
                        seq = torch.arange(n, device=device)
                        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                        alphas[i] = alpha
                        if alpha > 1:
                            Ds[i] = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))
                    
                    min_D_index = torch.argmin(Ds)
                    final_alpha = alphas[min_D_index]
                    final_D = Ds[min_D_index]
    
                # Convert to item() for storing results
                final_alpha = final_alpha.item()
                final_D = final_D.item()
                final_alphahat = final_alpha * math.log10(spectral_norm)
    
                results['spectral_norm'].append(spectral_norm)
                results['alphahat'].append(final_alphahat)
                results['norm'].append(fnorm)
                results['alpha'].append(final_alpha)
                results['D'].append(final_D)
                results['longname'].append(name)
                results['eigs'].append(nz_eigs.detach().cpu().numpy())
                results['eigs_num'].append(len(nz_eigs))
    
        return results
 
    def get_layer_temps(self, assign_func, metric_scores, untuned_lr):
        n = len(metric_scores)
        idx = [i for i in range(n)]
        temps = np.array([untuned_lr] * n)
        
        if assign_func == 'tb_linear_map':
            lr_range = [self.lr_min_ratio * untuned_lr,  self.lr_max_ratio * untuned_lr]
            score_range = [min(metric_scores),  max(metric_scores)]
            temps = np.interp(metric_scores, score_range, lr_range)

        elif assign_func == 'tb_sqrt':
            temps = np.sqrt(metric_scores)/np.sum(np.sqrt(metric_scores)) * n * untuned_lr

        elif assign_func == 'tb_log2':
            temps = np.log2(metric_scores)/np.sum(np.log2(metric_scores)) * n * untuned_lr

        elif assign_func == 'tb_step':
            idxes = np.argsort(metric_scores)
            unsort_temps = [untuned_lr * (self.lr_min_ratio + (self.lr_max_ratio - self.lr_min_ratio) * i / n) for i in range(n)]
            temps = [value for _, value in sorted(list(zip(idxes, unsort_temps)), key=itemgetter(0))]
        
        else:
            raise NotImplementedError
        
        return temps
    
    


# print('--------------------')
# longname_lst = []
# type_lst = []
# bn_to_conv = {}
# for name, module in self.net.named_modules():
#     if isinstance(module, nn.Conv2d):
#         longname_lst.append(name)
#         type_lst.append('nn.Conv2d')
#     if isinstance(module, nn.BatchNorm2d):
#         if 'nn.Conv2d' == type_lst[-1]:
#             bn_to_conv[name] = longname_lst[-1]
#         longname_lst.append(name)
#         type_lst.append('nn.BatchNorm2d')
# for key in bn_to_conv:
#     print(f"{key} -> {bn_to_conv[key]}")
# print('--------------------')