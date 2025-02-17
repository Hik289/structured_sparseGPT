# This file will contain helper functions related to the pruning process, including any specialized pruning functions and the SparseGPT functionality.
# DISCLAIMER: The SparseGPT class is a modified version of the original SparseGPT class. The original SparseGPT class can be found in [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot].

import math
import time

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from quant import *

# turned this flag to be True
DEBUG = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class SparseGPT_OPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.batch_inp = []
        self.batch_out = []

    def add_batch(self, inp, out, name, blocksize=1024):
        self.inp1 = inp
        #print(self.inp1.shape)
        self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        ###### added code
        if name == 'fc1' or name == 'fc2':
            self.batch_inp.append(inp[0].clone().detach())
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        ######
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        # del self.H 
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
            # print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))


    def simple_structured_prune(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=False, prune_cols=True):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows 
        and/or columns based on their importance measured by the L2 norm. The importance is defined as:
        - For rows: the L2 norm of each row.
        - For columns: the L2 norm of each column.
        The weights with the lowest importance, according to the given sparsity ratio, are set to zero.
        
        If a quantizer is available (i.e., self.quantizer exists), it will be initialized (if not already ready)
        and then used to quantize the pruned weights.
        
        Args:
            sparsity (float): The pruning ratio between 0 and 1 (e.g., 0.2 prunes 20% of the rows and/or columns).
            prune_rows (bool): Whether to prune rows (e.g., output neurons or convolution kernels).
            prune_cols (bool): Whether to prune columns (e.g., input features or connections).
        """
        # Clone the current layer's weights.
        W = self.layer.weight.data.clone()

        # Reshape weights if necessary.
        # For a Conv2d layer, flatten all dimensions except the first (output channels).
        if isinstance(self.layer, nn.Conv2d):
            original_shape = self.layer.weight.data.shape
            W = W.view(W.shape[0], -1)
        # For a transformers.Conv1D layer, assume a transpose is needed.
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = self.layer.weight.data.shape
            W = W.t()
        else:
            # For other layers with more than 2 dimensions, flatten from the second dimension onward.
            if W.dim() > 2:
                original_shape = self.layer.weight.data.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are of float type.
        W = W.float().to(self.layer.weight.dtype)
        num_rows, num_cols = W.shape

        # If a quantizer is present, ensure it is ready by finding quantization parameters.
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # Create a copy of the weight matrix for pruning.
        pruned_W = W.clone()
        if prune_rows:
            row_importance = torch.norm(W, p=2, dim=1)  # Shape: (num_rows,)       
            num_rows_to_prune = int(num_rows * sparsity) if prune_rows else 0   
            sorted_row_indices = torch.argsort(row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            pruned_W[rows_to_prune, :] = 0  # Zero out entire low-importance rows.
            print(f"Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows})")            
            
        if prune_cols:
            col_importance = torch.norm(W, p=2, dim=0)  # Shape: (num_cols,)
            num_cols_to_prune = int(num_cols * sparsity) if prune_cols else 0
            sorted_col_indices = torch.argsort(col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]        
            pruned_W[:, cols_to_prune] = 0  # Zero out entire low-importance columns.
            print(f"Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")
            
        # If a quantizer is available, quantize the pruned weights.
        if hasattr(self, 'quantizer'):
            # The quantize function is assumed to be defined externally.
            # Often, the quantize function might expect a certain shape.
            # Here we add an extra dimension if needed and then remove it afterward.
            pruned_W = pruned_W.unsqueeze(1)
            pruned_W = quantize(pruned_W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W.squeeze(1)

        # Restore the pruned weight matrix to its original shape.
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)

        # Update the layer's weight data with the pruned (and quantized) weight matrix.
        self.layer.weight.data = pruned_W.to(self.layer.weight.data.dtype)

        

    def simple_structured_prune_wanda(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)


        wanda_matrix = W.abs() * input_activation_norms.unsqueeze(0)

        pruned_W = W.clone()

        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        if prune_rows:
            # row_importance(i) = sum_j (|W_{i,j}| * input_activation_norms[j])
            wanda_row_importance = wanda_matrix.sum(dim=1)  # shape [num_rows]

            # How many rows to prune?
            num_rows_to_prune = int(num_rows * sparsity)
            # Sort rows by importance (ascending)
            sorted_row_indices = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            # Zero out the pruned rows
            pruned_W[rows_to_prune, :] = 0
            print(f"[Wanda] Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        # -----------------------------------------------
        # 4b. Column pruning by Wanda Score
        # -----------------------------------------------
        if prune_cols:
            # col_importance(j) = sum_i (|W_{i,j}| * input_activation_norms[j])
            # Notice input_activation_norms[j] is already factored in,
            # so effectively it's sum_i(|W[i,j]|) * input_activation_norms[j],
            # but we can just sum across rows from wanda_matrix.
            wanda_col_importance = wanda_matrix.sum(dim=0)  # shape [num_cols]

            # How many columns to prune?
            num_cols_to_prune = int(num_cols * sparsity)
            # Sort columns by importance (ascending)
            sorted_col_indices = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]
            # Zero out the pruned columns
            pruned_W[:, cols_to_prune] = 0
            print(f"[Wanda] Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)
    
    
    def simple_structured_prune_NIPE(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        #inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        #input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

            # -------------------------------------------------------------------------
        # 3. (Optional) Initialize or check quantizer
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # -------------------------------------------------------------------------
        # 4. Compute WaNDa importance scores
        #    Wanda score for each weight W[i, j]:  |W[i,j]| * input_activation_norms[j]
        #    Then sum across columns (row pruning) or sum across rows (col pruning).
        # -------------------------------------------------------------------------
        # Expand input_activation_norms to broadcast: shape [1, num_cols].
        # This line effectively does  W.abs() * input_activation_norms per entry:
        #print(W.shape,'111',self.inp1.shape)
        inp_reshaped = self.inp1.squeeze(0)  # 从 [1, 2048, 768] 变成 [2048, 768]
        wanda_matrix = inp_reshaped @ W.T  # [2048, 768] @ [768, 768]

        pruned_W = W.clone()
        #print(wanda_matrix.shape, W.shape)
        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        if prune_rows:
            # row_importance(i) = sum_j (|W_{i,j}| * input_activation_norms[j])
            wanda_row_importance = wanda_matrix.sum(dim=1).abs()  # shape [num_rows]

            # How many rows to prune?
            num_rows_to_prune = int(num_rows * sparsity)
            # Sort rows by importance (ascending)
            sorted_row_indices = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            # Zero out the pruned rows
            pruned_W[rows_to_prune,:] = 0
            print(f"[NIPE] Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        # -----------------------------------------------
        # 4b. Column pruning by Wanda Score
        # -----------------------------------------------
        if prune_cols:
            # col_importance(j) = sum_i (|W_{i,j}| * input_activation_norms[j])
            # Notice input_activation_norms[j] is already factored in,
            # so effectively it's sum_i(|W[i,j]|) * input_activation_norms[j],
            # but we can just sum across rows from wanda_matrix.
            wanda_col_importance = wanda_matrix.sum(dim=0).abs()  # shape [num_cols]
            #print(wanda_col_importance.shape)
            # How many columns to prune?
            num_cols_to_prune = int(num_cols * sparsity)
            # Sort columns by importance (ascending)
            sorted_col_indices = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]
            #print(cols_to_prune)
            # Zero out the pruned columns
            pruned_W[cols_to_prune, :] = 0
            print(f"[NIPE] Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)
    
    
    
    def simple_structured_iterative_correction_prune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        percdamp=128,
        blocksize=0.01,
        prune_rows=False,
        prune_cols=True,
        num_iter=10,
        lr=1e-3
    ):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows
        and/or columns based on their L2 norm importance. The least important rows (or columns)
        are set to zero according to the given sparsity ratio. (If a quantizer is available, it is applied.)
        
        Immediately after pruning, an iterative correction is performed that updates only the unpruned
        entries while the pruned entries are frozen (set to 0 and not updated). 
        ...
        """

        dev = getattr(self, "dev", self.layer.weight.device)  # device fallback

        # --- Structured Pruning ---
        # 1) Get the original weight, store if not stored yet
        W_full = self.layer.weight.detach().clone()  # safer than .data.clone()
        if not hasattr(self, 'orig_weight'):
            self.orig_weight = W_full.clone()

        # 2) Possibly reshape for Conv2d, etc.
        #    We'll keep W as the "2D view" if needed, but remember the original shape
        if isinstance(self.layer, nn.Conv2d):
            original_shape = W_full.shape  # e.g. (out_channels, in_channels, kh, kw)
            W = W_full.view(W_full.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W_full.shape
            W = W_full.t()  # e.g. for Conv1D in some HF models
        else:
            if W_full.dim() > 2:
                original_shape = W_full.shape
                W = W_full.view(W_full.shape[0], -1)
            else:
                original_shape = W_full.shape
                W = W_full
        
        W = W.float()
        num_rows, num_cols = W.shape
        
        # 3) If you have a quantizer, initialize it (not strictly needed here)
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # 4) Compute pruned weight and mask
        
        inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]


        wanda_matrix = W.abs() * input_activation_norms.unsqueeze(0)

        pruned_W = W.clone()

        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        
        mask = torch.ones_like(W, dtype=torch.bool)

        if prune_rows:
            wanda_row_importance = wanda_matrix.sum(dim=1) 
            num_rows_to_prune = int(num_rows * sparsity)
            sorted_rows = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_rows[:num_rows_to_prune]
            pruned_W[rows_to_prune, :] = 0
            mask[rows_to_prune, :] = False
            print(f"Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        if prune_cols:
            wanda_col_importance = wanda_matrix.sum(dim=0) 
            num_cols_to_prune = int(num_cols * sparsity)
            sorted_cols = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_cols[:num_cols_to_prune]
            pruned_W[:, cols_to_prune] = 0
            mask[:, cols_to_prune] = False
            print(f"Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # 5) If quantizer is available, apply it
        if hasattr(self, 'quantizer'):
            pruned_W_4q = pruned_W.unsqueeze(1)
            pruned_W_4q = quantize(pruned_W_4q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W_4q.squeeze(1)

        # 6) Restore shape for setting back to layer
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
            mask = mask.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
            
        # 7) Update the layer's weight (frozen) and store mask on correct device
        #    Since we don't want to track gradients on self.layer.weight,
        #    we can do a .detach() + copy_ or just assign data. 
        self.layer.weight.detach().copy_(pruned_W.to(self.layer.weight.dtype))
        self.layer.prune_mask = mask.to(dev)
        self.layer.weight.requires_grad = False  # 冻结layer原weight
        
        with torch.no_grad():
            if isinstance(self.layer, nn.Linear):
                in_features = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                #print(X.dtype, self.orig_weight.dtype, self.layer.bias.dtype)
                target = F.linear(X, self.orig_weight.to(self.layer.weight.dtype).to(dev), self.layer.bias)
            elif isinstance(self.layer, nn.Conv2d):
                in_channels = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                # Temporarily swap back the original weight to get the target
                current_weight = self.layer.weight.detach().clone()
                self.layer.weight.detach().copy_(self.orig_weight.to(self.layer.weight.dtype))
                target = self.layer(X)
                self.layer.weight.detach().copy_(current_weight)
            else:
                raise NotImplementedError("Iterative correction is only for Linear/Conv2d layers.")
                

        with torch.enable_grad():
            # --- Prepare the Trainable Parameter for Correction ---
            # We make a new Parameter that *does* require grad. 
            trainable_weight = nn.Parameter(pruned_W.clone().to(self.layer.weight.dtype), requires_grad=True)
            # Force pruned entries to zero right away
            trainable_weight.data *= self.layer.prune_mask.float().to(trainable_weight.device)

            # --- Iterative Correction ---
            print('Starting iterative correction ...')
            optimizer = torch.optim.SGD([trainable_weight], lr=1)

            # Synthetic input & target for matching the original layer's mapping


            # Perform gradient-based correction on unpruned entries
            # Make sure grad is enabled
            for i in range(num_iter):

                optimizer.zero_grad()
                # Masked effective weight
                effective_weight = trainable_weight * self.layer.prune_mask.float().to(trainable_weight.dtype).to(trainable_weight.device)
                #print(effective_weight.requires_grad, self.layer.bias.requires_grad)
                if isinstance(self.layer, nn.Linear):
                    output = F.linear(X, effective_weight, self.layer.bias)
                else:  # Conv2d
                    output = F.conv2d(
                        X, 
                        effective_weight, 
                        self.layer.bias,
                        stride=self.layer.stride,
                        padding=self.layer.padding,
                        dilation=self.layer.dilation,
                        groups=self.layer.groups
                    )

                loss = F.mse_loss(output, target)
                loss.backward()
                # grad = trainable_weight.grad
                # if grad is not None:
                #     print("grad min=%.6f, max=%.6f"%(grad.min(), grad.max()))
                # print("weight min=%.6f, max=%.6f"%(trainable_weight.data.min(), trainable_weight.data.max()))

                optimizer.step()
                    #print(output,target)
                    
                
                print(f"Iterative correction step {i+1}/{num_iter}, MSE loss: {loss.item():.6f}")

        # After correction, copy final masked trainable_weight back to the layer
        final_weight = trainable_weight.detach() * self.layer.prune_mask.float().to(trainable_weight.device)
        self.layer.weight.detach().copy_(final_weight.to(self.layer.weight.dtype))
        # Optionally freeze the layer's weight
        self.layer.weight.requires_grad = False

        print("Pruning + Correction complete.")


    def iterative_input_prune_gumbel(
            self,
            X: torch.Tensor,
            W: torch.Tensor,
            sparsity: float,
            prune_rows=False,
            prune_cols=True,
            num_iter: int = 100,
            lr: float = 1e-2,
            temperature: float = 1.0,   # Gumbel 温度
            hard_sample: bool = True,   # 是否做硬采样
            reg_coeff: float = 0.0,     # 用于稀疏正则的系数(可选)
        ):
            device = X.device
            N, D = X.shape

            # 原输出 (target)
            with torch.no_grad():
                XW = F.linear(X, W, self.layer.bias)  # [N, out_features]

            # 哪个维度要剪？
            if prune_cols:
                length = W.shape[1]  # 列数
            elif prune_rows:
                length = W.shape[0]  # 行数
            else:
                raise ValueError("Must set prune_rows or prune_cols to True.")

            # 你可以不强制 num_zero，但这里还是先算出理想的“目标1的个数”
            num_zero = int(sparsity * length)
            target_ones = length - num_zero

            # 1) 初始化可学习参数 theta
            theta_init = 0.01 * torch.randn(length, dtype=X.dtype)  # 可自行设计
            theta = nn.Parameter(theta_init.to(device), requires_grad=True)

            optimizer = torch.optim.SGD([theta], lr=1)

            for it in range(num_iter):
                optimizer.zero_grad()

                # 2) 采样 Gumbel 噪声 => Binary mask
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(theta) + 1e-20) + 1e-20).to(dtype = X.dtype)
                # z_i = (theta + gumbel_noise)/temperature
                z = (theta + gumbel_noise) / temperature
                mask_soft = torch.sigmoid(z)  # in (0,1)

                if hard_sample:
                    # Straight-Through: forward用hard, backward用soft
                    mask_hard = (mask_soft > 0.5).half()
                    mask_ste = mask_hard + (mask_soft - mask_soft.detach())
                else:
                    # 若不要硬采样，直接用soft值做mask
                    mask_ste = mask_soft

                # 3) 构造 W_pruned
                if prune_cols:
                    # mask => [col_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(0)
                else:
                    # mask => [row_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(1)

                # 4) 计算 MSE loss
                out_pruned = F.linear(X, W_pruned, self.layer.bias)
                loss_mse = F.mse_loss(out_pruned, XW)

                # 5) 如果想鼓励一定的稀疏度，可以加一个惩罚项
                #    例如 (mask的总和 - 目标值)^2，或用 L1, KL 等方式
                mask_sum = mask_ste.sum()
                loss_reg = reg_coeff * (mask_sum - target_ones).pow(2)

                loss = loss_mse + loss_reg
                loss.backward()
                optimizer.step()

                if (it + 1) % 10 == 0:
                    print(f"Iter {it+1}/{num_iter}, "
                        f"MSE={loss_mse.item():.6f}, "
                        f"MaskSum={mask_sum.item():.2f}")

            # 最后得到的 mask (可以根据 soft/hard 再做一次硬化)
            with torch.no_grad():
                final_mask = (torch.sigmoid(theta) > 0.5).float()

            return final_mask
        
   
    def iterative_input_prune_simplified(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        sparsity: float,
        prune_rows=False,
        prune_cols=True,
        num_iter: int = 1000,
        lr: float = 1e-2
    ):
        """
        在 (行/列) 维度对 W 做“离散剪枝掩码”并使用 Straight-Through Estimator 来更新 theta。
        但训练中不强制排序/置0, 只在结束后一次性地根据最终prob做硬剪枝。
        
        目标: 让 (X@W_pruned) ~ (X@W)。
        最终保留 (1-sparsity)*length 个最可能位置, 其余置0。
        """

        device = X.device
        N, D = X.shape

        # 1) 计算原输出 (不参与梯度)
        with torch.no_grad():
            XW = F.linear(X, W, self.layer.bias)  # [N, out_dim], 这里 out_dim==D 仅示例

        # 2) 判断剪枝维度
        if prune_cols:
            length = W.shape[1]  # 列数
        elif prune_rows:
            length = W.shape[0]  # 行数
        else:
            raise ValueError("Must set prune_rows or prune_cols to True.")

        # 3) 初始化可训练参数 theta
        with torch.enable_grad():
            theta = nn.Parameter(torch.zeros(length, dtype=X.dtype, device=device), requires_grad=True)
            optimizer = torch.optim.SGD([theta], lr=1)
            
            for it in range(num_iter):
                optimizer.zero_grad()

                # (a) 概率 prob in [0,1]
                prob = torch.sigmoid(theta)  # [length]

                # (c) 构造 W_pruned
                if prune_cols:
                    # mask => [length=col_dim], broadcast到 [row_dim, col_dim]
                    W_pruned = W * prob.unsqueeze(0)
                else:
                    # prune_rows
                    # mask => [length=row_dim], broadcast到 [row_dim, col_dim]
                    W_pruned = W * prob.unsqueeze(1)

                # (d) loss = MSE( X@W_pruned, XW )
                out_pruned = F.linear(X, W_pruned, self.layer.bias)
                loss = F.mse_loss(out_pruned, XW)

                loss.backward()
                optimizer.step()

                if (it+1) % 10 == 0 or it == 1:
                    print(f"Iter {it+1}/{num_iter}, Loss={loss.item():.6f}")

        # 4) 训练结束后 => 再次计算 prob = sigmoid(theta)，并做一次排序来剪枝
        with torch.no_grad():
            prob_final = torch.sigmoid(theta)
            # 按照prob从大到小排序, 保留 top-k
            k_to_keep = int((1 - sparsity) * length)
            sorted_idx = torch.argsort(prob_final, descending=True)  # 降序
            keep_idx = sorted_idx[:k_to_keep]
            
            final_mask = torch.zeros_like(prob_final)
            final_mask[keep_idx] = 1.0

        return final_mask 
   
    

    def iterative_input_prune_STE(self,
        X: torch.Tensor,
        W: torch.Tensor,
        sparsity: float,
        prune_rows=False,
        prune_cols=True,
        num_iter: int = 100,
        lr: float = 1e-2
    ):
        """
        在 (行/列) 维度对 W 做“离散剪枝掩码”并使用 Straight-Through Estimator 来更新 theta。
        目标是让 (X@W_pruned) ~ (X@W)。
        """
        device = X.device
        N, D = X.shape

        # 原输出
        with torch.no_grad():
            XW = F.linear(X, W, self.layer.bias)  # [N, D]

        # 哪个维度要剪？
        if prune_cols:
            length = W.shape[1]  # 列数
        elif prune_rows:
            length = W.shape[0]  # 行数
        else:
            raise ValueError("Must set prune_rows or prune_cols to True.")

        num_zero = int(sparsity * length)

        # 1) 初始化可训练参数 theta
        init_mask = torch.ones(length, dtype=X.dtype)
        init_mask[:num_zero] = 0.0
        init_mask = init_mask[torch.randperm(length)]
        theta_init = (init_mask * 2 - 1) + 0.1 * torch.randn(length,dtype=X.dtype)  # 负值对应0，正值对应1
        with torch.enable_grad():
            theta = nn.Parameter(theta_init.to(device), requires_grad=True)

            optimizer = torch.optim.SGD([theta], lr=0.1)

            for it in range(num_iter):
                optimizer.zero_grad()

                # (a) 计算概率 prob
                prob = torch.sigmoid(theta)  # [length], in [0,1]

                # (b) 硬剪枝：找最小的 num_zero 个令其=0，其余=1
                sorted_idx = torch.argsort(prob)  # 升序
                cutoff_idx = sorted_idx[:num_zero]

                mask_hard = torch.ones_like(prob,dtype=X.dtype)
                mask_hard[cutoff_idx] = 0.0  # 这里是离散操作 => 不可导

                # (c) 直通梯度 (STE) 核心： 让 forward 使用 mask_hard，backward 使用 prob
                #     mask_ste = mask_hard + (prob - prob.detach())
                #     这样 forward = mask_hard, backward d(mask_ste)/d(prob) = identity
                mask_ste = mask_hard + (prob - prob.detach())

                # (d) 构造 W_pruned
                if prune_cols:
                    # [row_dim, col_dim], mask => [col_dim]
                    W_pruned = W * mask_ste.unsqueeze(0)  # 广播到 [row_dim, col_dim]
                else:
                    # prune_rows
                    # mask => [row_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(1)

                # (e) loss = MSE(X @ W_pruned, XW)
                out_pruned = F.linear(X, W_pruned, self.layer.bias) # shape [N, D]
                loss = F.mse_loss(out_pruned, XW)

                loss.backward()
                optimizer.step()

                if (it+1) % 10 == 0 or it == 1:
                    print(f"Iter {it+1}/{num_iter}, Loss={loss.item():.6f}")

        # 训练完成后，再计算一次最终硬掩码
        with torch.no_grad():
            prob = torch.sigmoid(theta)
            sorted_idx = torch.argsort(prob)
            cutoff_idx = sorted_idx[:num_zero]

            final_mask = torch.ones_like(prob)
            final_mask[cutoff_idx] = 0.0

        return final_mask


    def prob_structured_iterative_correction_prune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        percdamp=128,
        blocksize=0.01,
        prune_rows=False,
        prune_cols=True,
        num_iter=100,
        lr=1e-2
    ):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows
        and/or columns based on their L2 norm importance. The least important rows (or columns)
        are set to zero according to the given sparsity ratio. (If a quantizer is available, it is applied.)
        
        Immediately after pruning, an iterative correction is performed that updates only the unpruned
        entries while the pruned entries are frozen (set to 0 and not updated). 
        ...
        """

        dev = getattr(self, "dev", self.layer.weight.device)  # device fallback

        # --- Structured Pruning ---
        # 1) Get the original weight, store if not stored yet
        W_full = self.layer.weight.detach().clone()  # safer than .data.clone()
        if not hasattr(self, 'orig_weight'):
            self.orig_weight = W_full.clone()

        # 2) Possibly reshape for Conv2d, etc.
        #    We'll keep W as the "2D view" if needed, but remember the original shape
        if isinstance(self.layer, nn.Conv2d):
            original_shape = W_full.shape  # e.g. (out_channels, in_channels, kh, kw)
            W = W_full.view(W_full.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W_full.shape
            W = W_full.t()  # e.g. for Conv1D in some HF models
        else:
            if W_full.dim() > 2:
                original_shape = W_full.shape
                W = W_full.view(W_full.shape[0], -1)
            else:
                original_shape = W_full.shape
                W = W_full
        
        W = W.float()
        num_rows, num_cols = W.shape
        
        # 3) If you have a quantizer, initialize it (not strictly needed here)
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # 4) Compute pruned weight and mask
        X = self.inp1.squeeze(0)
        W = W.to(X.dtype)
        with torch.enable_grad():
            print(X.shape,W.shape)
            mask = self.iterative_input_prune_simplified(X, W, sparsity, prune_rows,prune_cols, num_iter=100)

        # 应用到 X 上:
        pruned_W = W * mask.unsqueeze(0)
        
        

        # 5) If quantizer is available, apply it
        if hasattr(self, 'quantizer'):
            pruned_W_4q = pruned_W.unsqueeze(1)
            pruned_W_4q = quantize(pruned_W_4q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W_4q.squeeze(1)

        # 6) Restore shape for setting back to layer
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
            mask = mask.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
            
        # 7) Update the layer's weight (frozen) and store mask on correct device
        #    Since we don't want to track gradients on self.layer.weight,
        #    we can do a .detach() + copy_ or just assign data. 
        self.layer.weight.detach().copy_(pruned_W.to(self.layer.weight.dtype))
        self.layer.prune_mask = mask.to(dev)
        self.layer.weight.requires_grad = False  # 冻结layer原weight
        
        with torch.no_grad():
            if isinstance(self.layer, nn.Linear):
                in_features = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                #print(X.dtype, self.orig_weight.dtype, self.layer.bias.dtype)
                target = F.linear(X, self.orig_weight.to(self.layer.weight.dtype).to(dev), self.layer.bias)
            elif isinstance(self.layer, nn.Conv2d):
                in_channels = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                # Temporarily swap back the original weight to get the target
                current_weight = self.layer.weight.detach().clone()
                self.layer.weight.detach().copy_(self.orig_weight.to(self.layer.weight.dtype))
                target = self.layer(X)
                self.layer.weight.detach().copy_(current_weight)
            else:
                raise NotImplementedError("Iterative correction is only for Linear/Conv2d layers.")
                

        with torch.enable_grad():
            # --- Prepare the Trainable Parameter for Correction ---
            # We make a new Parameter that *does* require grad. 
            trainable_weight = nn.Parameter(pruned_W.clone().to(self.layer.weight.dtype), requires_grad=True)
            # Force pruned entries to zero right away
            trainable_weight.data *= self.layer.prune_mask.float().to(trainable_weight.device)

            # --- Iterative Correction ---
            print('Starting iterative correction ...')
            optimizer = torch.optim.SGD([trainable_weight], lr=1)

            # Synthetic input & target for matching the original layer's mapping


            # Perform gradient-based correction on unpruned entries
            # Make sure grad is enabled
            for i in range(num_iter):

                optimizer.zero_grad()
                # Masked effective weight
                effective_weight = trainable_weight * self.layer.prune_mask.float().to(trainable_weight.dtype).to(trainable_weight.device)
                #print(effective_weight.requires_grad, self.layer.bias.requires_grad)
                if isinstance(self.layer, nn.Linear):
                    output = F.linear(X, effective_weight, self.layer.bias)
                else:  # Conv2d
                    output = F.conv2d(
                        X, 
                        effective_weight, 
                        self.layer.bias,
                        stride=self.layer.stride,
                        padding=self.layer.padding,
                        dilation=self.layer.dilation,
                        groups=self.layer.groups
                    )

                loss = F.mse_loss(output, target)
                loss.backward()
                # grad = trainable_weight.grad
                # if grad is not None:
                #     print("grad min=%.6f, max=%.6f"%(grad.min(), grad.max()))
                # print("weight min=%.6f, max=%.6f"%(trainable_weight.data.min(), trainable_weight.data.max()))

                optimizer.step()
                    #print(output,target)
                    
                
                print(f"Iterative correction step {i+1}/{num_iter}, MSE loss: {loss.item():.6f}")

        # After correction, copy final masked trainable_weight back to the layer
        final_weight = trainable_weight.detach() * self.layer.prune_mask.float().to(trainable_weight.device)
        self.layer.weight.detach().copy_(final_weight.to(self.layer.weight.dtype))
        # Optionally freeze the layer's weight
        self.layer.weight.requires_grad = False

        print("Pruning + Correction complete.")

    
    def simple_structured_prune_probe(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        #inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        #input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

            # -------------------------------------------------------------------------
        # 3. (Optional) Initialize or check quantizer
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        X = self.inp1.squeeze(0)
        W = W.to(X.dtype)
        with torch.enable_grad():
            print(X.shape,W.shape)
            mask = self.iterative_input_prune_STE(X, W, sparsity, prune_rows,prune_cols, num_iter=100)

        # 应用到 X 上:
        pruned_W = W * mask.unsqueeze(0)

        pruned_W = W.clone()
        # -----------------------------------------------

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)

    


    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()




class SparseGPT_LlaMA:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.batch_inp = []
        self.batch_out = []

    def add_batch(self, inp, out, name, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        ###### added code
        if name == 'mlp.up_proj' or name == 'mlp.down_proj':
            self.batch_inp.append(inp[0].clone().detach())
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        if name == 'mlp.gate_proj':   # for this layer, we only store the outputs. for inputs, they are shared with 'mlp.up_proj'
            if len(out.shape) == 3:
                out = out.squeeze(0)
            self.batch_out.append(out.clone().detach())
        ######
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        # del self.H 
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if DEBUG:
            #     self.layer.weight.data[:, :i2] = W[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #     print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # if DEBUG:
            # print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    def simple_structured_prune(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=False, prune_cols=True):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows 
        and/or columns based on their importance measured by the L2 norm. The importance is defined as:
        - For rows: the L2 norm of each row.
        - For columns: the L2 norm of each column.
        The weights with the lowest importance, according to the given sparsity ratio, are set to zero.
        
        If a quantizer is available (i.e., self.quantizer exists), it will be initialized (if not already ready)
        and then used to quantize the pruned weights.
        
        Args:
            sparsity (float): The pruning ratio between 0 and 1 (e.g., 0.2 prunes 20% of the rows and/or columns).
            prune_rows (bool): Whether to prune rows (e.g., output neurons or convolution kernels).
            prune_cols (bool): Whether to prune columns (e.g., input features or connections).
        """
        # Clone the current layer's weights.
        W = self.layer.weight.data.clone()

        # Reshape weights if necessary.
        # For a Conv2d layer, flatten all dimensions except the first (output channels).
        if isinstance(self.layer, nn.Conv2d):
            original_shape = self.layer.weight.data.shape
            W = W.view(W.shape[0], -1)
        # For a transformers.Conv1D layer, assume a transpose is needed.
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = self.layer.weight.data.shape
            W = W.t()
        else:
            # For other layers with more than 2 dimensions, flatten from the second dimension onward.
            if W.dim() > 2:
                original_shape = self.layer.weight.data.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are of float type.
        W = W.float().to(self.layer.weight.dtype)
        num_rows, num_cols = W.shape

        # If a quantizer is present, ensure it is ready by finding quantization parameters.
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # Create a copy of the weight matrix for pruning.
        pruned_W = W.clone()
        if prune_rows:
            row_importance = torch.norm(W, p=2, dim=1)  # Shape: (num_rows,)       
            num_rows_to_prune = int(num_rows * sparsity) if prune_rows else 0   
            sorted_row_indices = torch.argsort(row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            pruned_W[rows_to_prune, :] = 0  # Zero out entire low-importance rows.
            print(f"Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows})")            
            
        if prune_cols:
            col_importance = torch.norm(W, p=2, dim=0)  # Shape: (num_cols,)
            num_cols_to_prune = int(num_cols * sparsity) if prune_cols else 0
            sorted_col_indices = torch.argsort(col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]        
            pruned_W[:, cols_to_prune] = 0  # Zero out entire low-importance columns.
            print(f"Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")
            
        # If a quantizer is available, quantize the pruned weights.
        if hasattr(self, 'quantizer'):
            # The quantize function is assumed to be defined externally.
            # Often, the quantize function might expect a certain shape.
            # Here we add an extra dimension if needed and then remove it afterward.
            pruned_W = pruned_W.unsqueeze(1)
            pruned_W = quantize(pruned_W, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W.squeeze(1)

        # Restore the pruned weight matrix to its original shape.
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)

        # Update the layer's weight data with the pruned (and quantized) weight matrix.
        self.layer.weight.data = pruned_W.to(self.layer.weight.data.dtype)

        

    def simple_structured_prune_wanda(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)


        wanda_matrix = W.abs() * input_activation_norms.unsqueeze(0)

        pruned_W = W.clone()

        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        if prune_rows:
            # row_importance(i) = sum_j (|W_{i,j}| * input_activation_norms[j])
            wanda_row_importance = wanda_matrix.sum(dim=1)  # shape [num_rows]

            # How many rows to prune?
            num_rows_to_prune = int(num_rows * sparsity)
            # Sort rows by importance (ascending)
            sorted_row_indices = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            # Zero out the pruned rows
            pruned_W[rows_to_prune, :] = 0
            print(f"[Wanda] Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        # -----------------------------------------------
        # 4b. Column pruning by Wanda Score
        # -----------------------------------------------
        if prune_cols:
            # col_importance(j) = sum_i (|W_{i,j}| * input_activation_norms[j])
            # Notice input_activation_norms[j] is already factored in,
            # so effectively it's sum_i(|W[i,j]|) * input_activation_norms[j],
            # but we can just sum across rows from wanda_matrix.
            wanda_col_importance = wanda_matrix.sum(dim=0)  # shape [num_cols]

            # How many columns to prune?
            num_cols_to_prune = int(num_cols * sparsity)
            # Sort columns by importance (ascending)
            sorted_col_indices = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]
            # Zero out the pruned columns
            pruned_W[:, cols_to_prune] = 0
            print(f"[Wanda] Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)
    
    
    def simple_structured_prune_NIPE(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        #inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        #input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

            # -------------------------------------------------------------------------
        # 3. (Optional) Initialize or check quantizer
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # -------------------------------------------------------------------------
        # 4. Compute WaNDa importance scores
        #    Wanda score for each weight W[i, j]:  |W[i,j]| * input_activation_norms[j]
        #    Then sum across columns (row pruning) or sum across rows (col pruning).
        # -------------------------------------------------------------------------
        # Expand input_activation_norms to broadcast: shape [1, num_cols].
        # This line effectively does  W.abs() * input_activation_norms per entry:
        #print(W.shape,'111',self.inp1.shape)
        inp_reshaped = self.inp1.squeeze(0)  # 从 [1, 2048, 768] 变成 [2048, 768]
        wanda_matrix = inp_reshaped @ W.T  # [2048, 768] @ [768, 768]

        pruned_W = W.clone()
        #print(wanda_matrix.shape, W.shape)
        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        if prune_rows:
            # row_importance(i) = sum_j (|W_{i,j}| * input_activation_norms[j])
            wanda_row_importance = wanda_matrix.sum(dim=1).abs()  # shape [num_rows]

            # How many rows to prune?
            num_rows_to_prune = int(num_rows * sparsity)
            # Sort rows by importance (ascending)
            sorted_row_indices = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_row_indices[:num_rows_to_prune]
            # Zero out the pruned rows
            pruned_W[rows_to_prune,:] = 0
            print(f"[NIPE] Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        # -----------------------------------------------
        # 4b. Column pruning by Wanda Score
        # -----------------------------------------------
        if prune_cols:
            # col_importance(j) = sum_i (|W_{i,j}| * input_activation_norms[j])
            # Notice input_activation_norms[j] is already factored in,
            # so effectively it's sum_i(|W[i,j]|) * input_activation_norms[j],
            # but we can just sum across rows from wanda_matrix.
            wanda_col_importance = wanda_matrix.sum(dim=0).abs()  # shape [num_cols]
            #print(wanda_col_importance.shape)
            # How many columns to prune?
            num_cols_to_prune = int(num_cols * sparsity)
            # Sort columns by importance (ascending)
            sorted_col_indices = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_col_indices[:num_cols_to_prune]
            #print(cols_to_prune)
            # Zero out the pruned columns
            pruned_W[cols_to_prune, :] = 0
            print(f"[NIPE] Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)
    
    
    
    def simple_structured_iterative_correction_prune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        percdamp=128,
        blocksize=0.01,
        prune_rows=False,
        prune_cols=True,
        num_iter=10,
        lr=1e-3
    ):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows
        and/or columns based on their L2 norm importance. The least important rows (or columns)
        are set to zero according to the given sparsity ratio. (If a quantizer is available, it is applied.)
        
        Immediately after pruning, an iterative correction is performed that updates only the unpruned
        entries while the pruned entries are frozen (set to 0 and not updated). 
        ...
        """

        dev = getattr(self, "dev", self.layer.weight.device)  # device fallback

        # --- Structured Pruning ---
        # 1) Get the original weight, store if not stored yet
        W_full = self.layer.weight.detach().clone()  # safer than .data.clone()
        if not hasattr(self, 'orig_weight'):
            self.orig_weight = W_full.clone()

        # 2) Possibly reshape for Conv2d, etc.
        #    We'll keep W as the "2D view" if needed, but remember the original shape
        if isinstance(self.layer, nn.Conv2d):
            original_shape = W_full.shape  # e.g. (out_channels, in_channels, kh, kw)
            W = W_full.view(W_full.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W_full.shape
            W = W_full.t()  # e.g. for Conv1D in some HF models
        else:
            if W_full.dim() > 2:
                original_shape = W_full.shape
                W = W_full.view(W_full.shape[0], -1)
            else:
                original_shape = W_full.shape
                W = W_full
        
        W = W.float()
        num_rows, num_cols = W.shape
        
        # 3) If you have a quantizer, initialize it (not strictly needed here)
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # 4) Compute pruned weight and mask
        
        inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]


        wanda_matrix = W.abs() * input_activation_norms.unsqueeze(0)

        pruned_W = W.clone()

        # -----------------------------------------------
        # 4a. Row pruning by Wanda Score
        # -----------------------------------------------
        
        mask = torch.ones_like(W, dtype=torch.bool)

        if prune_rows:
            wanda_row_importance = wanda_matrix.sum(dim=1) 
            num_rows_to_prune = int(num_rows * sparsity)
            sorted_rows = torch.argsort(wanda_row_importance)
            rows_to_prune = sorted_rows[:num_rows_to_prune]
            pruned_W[rows_to_prune, :] = 0
            mask[rows_to_prune, :] = False
            print(f"Pruning completed: pruned {num_rows_to_prune} rows (out of {num_rows}).")

        if prune_cols:
            wanda_col_importance = wanda_matrix.sum(dim=0) 
            num_cols_to_prune = int(num_cols * sparsity)
            sorted_cols = torch.argsort(wanda_col_importance)
            cols_to_prune = sorted_cols[:num_cols_to_prune]
            pruned_W[:, cols_to_prune] = 0
            mask[:, cols_to_prune] = False
            print(f"Pruning completed: pruned {num_cols_to_prune} columns (out of {num_cols}).")

        # 5) If quantizer is available, apply it
        if hasattr(self, 'quantizer'):
            pruned_W_4q = pruned_W.unsqueeze(1)
            pruned_W_4q = quantize(pruned_W_4q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W_4q.squeeze(1)

        # 6) Restore shape for setting back to layer
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
            mask = mask.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
            
        # 7) Update the layer's weight (frozen) and store mask on correct device
        #    Since we don't want to track gradients on self.layer.weight,
        #    we can do a .detach() + copy_ or just assign data. 
        self.layer.weight.detach().copy_(pruned_W.to(self.layer.weight.dtype))
        self.layer.prune_mask = mask.to(dev)
        self.layer.weight.requires_grad = False  # 冻结layer原weight
        
        with torch.no_grad():
            if isinstance(self.layer, nn.Linear):
                in_features = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                #print(X.dtype, self.orig_weight.dtype, self.layer.bias.dtype)
                target = F.linear(X, self.orig_weight.to(self.layer.weight.dtype).to(dev), self.layer.bias)
            elif isinstance(self.layer, nn.Conv2d):
                in_channels = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                # Temporarily swap back the original weight to get the target
                current_weight = self.layer.weight.detach().clone()
                self.layer.weight.detach().copy_(self.orig_weight.to(self.layer.weight.dtype))
                target = self.layer(X)
                self.layer.weight.detach().copy_(current_weight)
            else:
                raise NotImplementedError("Iterative correction is only for Linear/Conv2d layers.")
                

        with torch.enable_grad():
            # --- Prepare the Trainable Parameter for Correction ---
            # We make a new Parameter that *does* require grad. 
            trainable_weight = nn.Parameter(pruned_W.clone().to(self.layer.weight.dtype), requires_grad=True)
            # Force pruned entries to zero right away
            trainable_weight.data *= self.layer.prune_mask.float().to(trainable_weight.device)

            # --- Iterative Correction ---
            print('Starting iterative correction ...')
            optimizer = torch.optim.SGD([trainable_weight], lr=1)

            # Synthetic input & target for matching the original layer's mapping


            # Perform gradient-based correction on unpruned entries
            # Make sure grad is enabled
            for i in range(num_iter):

                optimizer.zero_grad()
                # Masked effective weight
                effective_weight = trainable_weight * self.layer.prune_mask.float().to(trainable_weight.dtype).to(trainable_weight.device)
                #print(effective_weight.requires_grad, self.layer.bias.requires_grad)
                if isinstance(self.layer, nn.Linear):
                    output = F.linear(X, effective_weight, self.layer.bias)
                else:  # Conv2d
                    output = F.conv2d(
                        X, 
                        effective_weight, 
                        self.layer.bias,
                        stride=self.layer.stride,
                        padding=self.layer.padding,
                        dilation=self.layer.dilation,
                        groups=self.layer.groups
                    )

                loss = F.mse_loss(output, target)
                loss.backward()
                # grad = trainable_weight.grad
                # if grad is not None:
                #     print("grad min=%.6f, max=%.6f"%(grad.min(), grad.max()))
                # print("weight min=%.6f, max=%.6f"%(trainable_weight.data.min(), trainable_weight.data.max()))

                optimizer.step()
                    #print(output,target)
                    
                
                print(f"Iterative correction step {i+1}/{num_iter}, MSE loss: {loss.item():.6f}")

        # After correction, copy final masked trainable_weight back to the layer
        final_weight = trainable_weight.detach() * self.layer.prune_mask.float().to(trainable_weight.device)
        self.layer.weight.detach().copy_(final_weight.to(self.layer.weight.dtype))
        # Optionally freeze the layer's weight
        self.layer.weight.requires_grad = False

        print("Pruning + Correction complete.")


    def iterative_input_prune_gumbel(
            self,
            X: torch.Tensor,
            W: torch.Tensor,
            sparsity: float,
            prune_rows=False,
            prune_cols=True,
            num_iter: int = 100,
            lr: float = 1e-2,
            temperature: float = 1.0,   # Gumbel 温度
            hard_sample: bool = True,   # 是否做硬采样
            reg_coeff: float = 0.0,     # 用于稀疏正则的系数(可选)
        ):
            device = X.device
            N, D = X.shape

            # 原输出 (target)
            with torch.no_grad():
                XW = F.linear(X, W, self.layer.bias)  # [N, out_features]

            # 哪个维度要剪？
            if prune_cols:
                length = W.shape[1]  # 列数
            elif prune_rows:
                length = W.shape[0]  # 行数
            else:
                raise ValueError("Must set prune_rows or prune_cols to True.")

            # 你可以不强制 num_zero，但这里还是先算出理想的“目标1的个数”
            num_zero = int(sparsity * length)
            target_ones = length - num_zero

            # 1) 初始化可学习参数 theta
            theta_init = 0.01 * torch.randn(length, dtype=X.dtype)  # 可自行设计
            theta = nn.Parameter(theta_init.to(device), requires_grad=True)

            optimizer = torch.optim.SGD([theta], lr=1)

            for it in range(num_iter):
                optimizer.zero_grad()

                # 2) 采样 Gumbel 噪声 => Binary mask
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(theta) + 1e-20) + 1e-20).to(dtype = X.dtype)
                # z_i = (theta + gumbel_noise)/temperature
                z = (theta + gumbel_noise) / temperature
                mask_soft = torch.sigmoid(z)  # in (0,1)

                if hard_sample:
                    # Straight-Through: forward用hard, backward用soft
                    mask_hard = (mask_soft > 0.5).half()
                    mask_ste = mask_hard + (mask_soft - mask_soft.detach())
                else:
                    # 若不要硬采样，直接用soft值做mask
                    mask_ste = mask_soft

                # 3) 构造 W_pruned
                if prune_cols:
                    # mask => [col_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(0)
                else:
                    # mask => [row_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(1)

                # 4) 计算 MSE loss
                out_pruned = F.linear(X, W_pruned, self.layer.bias)
                loss_mse = F.mse_loss(out_pruned, XW)

                # 5) 如果想鼓励一定的稀疏度，可以加一个惩罚项
                #    例如 (mask的总和 - 目标值)^2，或用 L1, KL 等方式
                mask_sum = mask_ste.sum()
                loss_reg = reg_coeff * (mask_sum - target_ones).pow(2)

                loss = loss_mse + loss_reg
                loss.backward()
                optimizer.step()

                if (it + 1) % 10 == 0:
                    print(f"Iter {it+1}/{num_iter}, "
                        f"MSE={loss_mse.item():.6f}, "
                        f"MaskSum={mask_sum.item():.2f}")

            # 最后得到的 mask (可以根据 soft/hard 再做一次硬化)
            with torch.no_grad():
                final_mask = (torch.sigmoid(theta) > 0.5).float()

            return final_mask
        
   
    def iterative_input_prune_simplified(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        sparsity: float,
        prune_rows=False,
        prune_cols=True,
        num_iter: int = 1000,
        lr: float = 1e-2
    ):
        """
        在 (行/列) 维度对 W 做“离散剪枝掩码”并使用 Straight-Through Estimator 来更新 theta。
        但训练中不强制排序/置0, 只在结束后一次性地根据最终prob做硬剪枝。
        
        目标: 让 (X@W_pruned) ~ (X@W)。
        最终保留 (1-sparsity)*length 个最可能位置, 其余置0。
        """

        device = X.device
        N, D = X.shape

        # 1) 计算原输出 (不参与梯度)
        with torch.no_grad():
            XW = F.linear(X, W, self.layer.bias)  # [N, out_dim], 这里 out_dim==D 仅示例

        # 2) 判断剪枝维度
        if prune_cols:
            length = W.shape[1]  # 列数
        elif prune_rows:
            length = W.shape[0]  # 行数
        else:
            raise ValueError("Must set prune_rows or prune_cols to True.")

        # 3) 初始化可训练参数 theta
        with torch.enable_grad():
            theta = nn.Parameter(torch.zeros(length, dtype=X.dtype, device=device), requires_grad=True)
            optimizer = torch.optim.SGD([theta], lr=1)
            
            for it in range(num_iter):
                optimizer.zero_grad()

                # (a) 概率 prob in [0,1]
                prob = torch.sigmoid(theta)  # [length]

                # (c) 构造 W_pruned
                if prune_cols:
                    # mask => [length=col_dim], broadcast到 [row_dim, col_dim]
                    W_pruned = W * prob.unsqueeze(0)
                else:
                    # prune_rows
                    # mask => [length=row_dim], broadcast到 [row_dim, col_dim]
                    W_pruned = W * prob.unsqueeze(1)

                # (d) loss = MSE( X@W_pruned, XW )
                out_pruned = F.linear(X, W_pruned, self.layer.bias)
                loss = F.mse_loss(out_pruned, XW)

                loss.backward()
                optimizer.step()

                if (it+1) % 10 == 0 or it == 1:
                    print(f"Iter {it+1}/{num_iter}, Loss={loss.item():.6f}")

        # 4) 训练结束后 => 再次计算 prob = sigmoid(theta)，并做一次排序来剪枝
        with torch.no_grad():
            prob_final = torch.sigmoid(theta)
            # 按照prob从大到小排序, 保留 top-k
            k_to_keep = int((1 - sparsity) * length)
            sorted_idx = torch.argsort(prob_final, descending=True)  # 降序
            keep_idx = sorted_idx[:k_to_keep]
            
            final_mask = torch.zeros_like(prob_final)
            final_mask[keep_idx] = 1.0

        return final_mask 
   
    

    def iterative_input_prune_STE(self,
        X: torch.Tensor,
        W: torch.Tensor,
        sparsity: float,
        prune_rows=False,
        prune_cols=True,
        num_iter: int = 100,
        lr: float = 1e-2
    ):
        """
        在 (行/列) 维度对 W 做“离散剪枝掩码”并使用 Straight-Through Estimator 来更新 theta。
        目标是让 (X@W_pruned) ~ (X@W)。
        """
        device = X.device
        N, D = X.shape

        # 原输出
        with torch.no_grad():
            XW = F.linear(X, W, self.layer.bias)  # [N, D]

        # 哪个维度要剪？
        if prune_cols:
            length = W.shape[1]  # 列数
        elif prune_rows:
            length = W.shape[0]  # 行数
        else:
            raise ValueError("Must set prune_rows or prune_cols to True.")

        num_zero = int(sparsity * length)

        # 1) 初始化可训练参数 theta
        init_mask = torch.ones(length, dtype=X.dtype)
        init_mask[:num_zero] = 0.0
        init_mask = init_mask[torch.randperm(length)]
        theta_init = (init_mask * 2 - 1) + 0.1 * torch.randn(length,dtype=X.dtype)  # 负值对应0，正值对应1
        with torch.enable_grad():
            theta = nn.Parameter(theta_init.to(device), requires_grad=True)

            optimizer = torch.optim.SGD([theta], lr=0.1)

            for it in range(num_iter):
                optimizer.zero_grad()

                # (a) 计算概率 prob
                prob = torch.sigmoid(theta)  # [length], in [0,1]

                # (b) 硬剪枝：找最小的 num_zero 个令其=0，其余=1
                sorted_idx = torch.argsort(prob)  # 升序
                cutoff_idx = sorted_idx[:num_zero]

                mask_hard = torch.ones_like(prob,dtype=X.dtype)
                mask_hard[cutoff_idx] = 0.0  # 这里是离散操作 => 不可导

                # (c) 直通梯度 (STE) 核心： 让 forward 使用 mask_hard，backward 使用 prob
                #     mask_ste = mask_hard + (prob - prob.detach())
                #     这样 forward = mask_hard, backward d(mask_ste)/d(prob) = identity
                mask_ste = mask_hard + (prob - prob.detach())

                # (d) 构造 W_pruned
                if prune_cols:
                    # [row_dim, col_dim], mask => [col_dim]
                    W_pruned = W * mask_ste.unsqueeze(0)  # 广播到 [row_dim, col_dim]
                else:
                    # prune_rows
                    # mask => [row_dim], 广播到 [row_dim, col_dim]
                    W_pruned = W * mask_ste.unsqueeze(1)

                # (e) loss = MSE(X @ W_pruned, XW)
                out_pruned = F.linear(X, W_pruned, self.layer.bias) # shape [N, D]
                loss = F.mse_loss(out_pruned, XW)

                loss.backward()
                optimizer.step()

                if (it+1) % 10 == 0 or it == 1:
                    print(f"Iter {it+1}/{num_iter}, Loss={loss.item():.6f}")

        # 训练完成后，再计算一次最终硬掩码
        with torch.no_grad():
            prob = torch.sigmoid(theta)
            sorted_idx = torch.argsort(prob)
            cutoff_idx = sorted_idx[:num_zero]

            final_mask = torch.ones_like(prob)
            final_mask[cutoff_idx] = 0.0

        return final_mask


    def prob_structured_iterative_correction_prune(
        self,
        sparsity,
        prunen=0,
        prunem=0,
        percdamp=128,
        blocksize=0.01,
        prune_rows=False,
        prune_cols=True,
        num_iter=100,
        lr=1e-2
    ):
        """
        Performs simple structured pruning on the current layer's weights by removing entire rows
        and/or columns based on their L2 norm importance. The least important rows (or columns)
        are set to zero according to the given sparsity ratio. (If a quantizer is available, it is applied.)
        
        Immediately after pruning, an iterative correction is performed that updates only the unpruned
        entries while the pruned entries are frozen (set to 0 and not updated). 
        ...
        """

        dev = getattr(self, "dev", self.layer.weight.device)  # device fallback

        # --- Structured Pruning ---
        # 1) Get the original weight, store if not stored yet
        W_full = self.layer.weight.detach().clone()  # safer than .data.clone()
        if not hasattr(self, 'orig_weight'):
            self.orig_weight = W_full.clone()

        # 2) Possibly reshape for Conv2d, etc.
        #    We'll keep W as the "2D view" if needed, but remember the original shape
        if isinstance(self.layer, nn.Conv2d):
            original_shape = W_full.shape  # e.g. (out_channels, in_channels, kh, kw)
            W = W_full.view(W_full.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W_full.shape
            W = W_full.t()  # e.g. for Conv1D in some HF models
        else:
            if W_full.dim() > 2:
                original_shape = W_full.shape
                W = W_full.view(W_full.shape[0], -1)
            else:
                original_shape = W_full.shape
                W = W_full
        
        W = W.float()
        num_rows, num_cols = W.shape
        
        # 3) If you have a quantizer, initialize it (not strictly needed here)
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        # 4) Compute pruned weight and mask
        X = self.inp1.squeeze(0)
        W = W.to(X.dtype)
        with torch.enable_grad():
            print(X.shape,W.shape)
            mask = self.iterative_input_prune_simplified(X, W, sparsity, prune_rows,prune_cols, num_iter=100)

        # 应用到 X 上:
        pruned_W = W * mask.unsqueeze(0)
        
        

        # 5) If quantizer is available, apply it
        if hasattr(self, 'quantizer'):
            pruned_W_4q = pruned_W.unsqueeze(1)
            pruned_W_4q = quantize(pruned_W_4q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W = pruned_W_4q.squeeze(1)

        # 6) Restore shape for setting back to layer
        if isinstance(self.layer, nn.Conv2d):
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            pruned_W = pruned_W.t().view(original_shape)
            mask = mask.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W = pruned_W.view(original_shape)
            mask = mask.view(original_shape)
            
        # 7) Update the layer's weight (frozen) and store mask on correct device
        #    Since we don't want to track gradients on self.layer.weight,
        #    we can do a .detach() + copy_ or just assign data. 
        self.layer.weight.detach().copy_(pruned_W.to(self.layer.weight.dtype))
        self.layer.prune_mask = mask.to(dev)
        self.layer.weight.requires_grad = False  # 冻结layer原weight
        
        with torch.no_grad():
            if isinstance(self.layer, nn.Linear):
                in_features = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                #print(X.dtype, self.orig_weight.dtype, self.layer.bias.dtype)
                target = F.linear(X, self.orig_weight.to(self.layer.weight.dtype).to(dev), self.layer.bias)
            elif isinstance(self.layer, nn.Conv2d):
                in_channels = self.layer.weight.shape[1]
                X = self.inp1.unsqueeze(0)
                # Temporarily swap back the original weight to get the target
                current_weight = self.layer.weight.detach().clone()
                self.layer.weight.detach().copy_(self.orig_weight.to(self.layer.weight.dtype))
                target = self.layer(X)
                self.layer.weight.detach().copy_(current_weight)
            else:
                raise NotImplementedError("Iterative correction is only for Linear/Conv2d layers.")
                

        with torch.enable_grad():
            # --- Prepare the Trainable Parameter for Correction ---
            # We make a new Parameter that *does* require grad. 
            trainable_weight = nn.Parameter(pruned_W.clone().to(self.layer.weight.dtype), requires_grad=True)
            # Force pruned entries to zero right away
            trainable_weight.data *= self.layer.prune_mask.float().to(trainable_weight.device)

            # --- Iterative Correction ---
            print('Starting iterative correction ...')
            optimizer = torch.optim.SGD([trainable_weight], lr=1)

            # Synthetic input & target for matching the original layer's mapping


            # Perform gradient-based correction on unpruned entries
            # Make sure grad is enabled
            for i in range(num_iter):

                optimizer.zero_grad()
                # Masked effective weight
                effective_weight = trainable_weight * self.layer.prune_mask.float().to(trainable_weight.dtype).to(trainable_weight.device)
                #print(effective_weight.requires_grad, self.layer.bias.requires_grad)
                if isinstance(self.layer, nn.Linear):
                    output = F.linear(X, effective_weight, self.layer.bias)
                else:  # Conv2d
                    output = F.conv2d(
                        X, 
                        effective_weight, 
                        self.layer.bias,
                        stride=self.layer.stride,
                        padding=self.layer.padding,
                        dilation=self.layer.dilation,
                        groups=self.layer.groups
                    )

                loss = F.mse_loss(output, target)
                loss.backward()
                # grad = trainable_weight.grad
                # if grad is not None:
                #     print("grad min=%.6f, max=%.6f"%(grad.min(), grad.max()))
                # print("weight min=%.6f, max=%.6f"%(trainable_weight.data.min(), trainable_weight.data.max()))

                optimizer.step()
                    #print(output,target)
                    
                
                print(f"Iterative correction step {i+1}/{num_iter}, MSE loss: {loss.item():.6f}")

        # After correction, copy final masked trainable_weight back to the layer
        final_weight = trainable_weight.detach() * self.layer.prune_mask.float().to(trainable_weight.device)
        self.layer.weight.detach().copy_(final_weight.to(self.layer.weight.dtype))
        # Optionally freeze the layer's weight
        self.layer.weight.requires_grad = False

        print("Pruning + Correction complete.")

    
    def simple_structured_prune_probe(self, sparsity, prunen=0, prunem=0, percdamp=128, blocksize=0.01,prune_rows=True, prune_cols=False):
        """
        Performs simple structured pruning on the current layer's weights using the WaNDa score:
            Wanda score: I(W) = |W| * (1 * ||X_in||^T)

        Args:
            sparsity (float): The fraction of rows/columns to prune (e.g., 0.2 means 20%).
            input_activation_norms (torch.Tensor): A 1D tensor of length (in_features) giving
                                                the L2 norm of each input dimension. 
                                                E.g., shape [num_cols].
            prune_rows (bool): Whether to prune entire rows.
            prune_cols (bool): Whether to prune entire columns.
        """

        # -------------------------------------------------------------------------
        # 1. Fetch and reshape weights
        # -------------------------------------------------------------------------
        W = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            original_shape = W.shape
            # Flatten out all dimensions except the first (output channels)
            W = W.view(W.shape[0], -1)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            original_shape = W.shape
            W = W.t()
        else:
            if W.dim() > 2:
                original_shape = W.shape
                W = W.view(W.shape[0], -1)
            else:
                original_shape = W.shape

        # Ensure the weights are in float (matching layer dtype is optional)
        W = W.float().to(self.layer.weight.dtype)

        num_rows, num_cols = W.shape


        #inp_reshaped = self.inp1.view(-1, self.inp1.shape[-1])       # shape [2048, 768]
        #input_activation_norms = inp_reshaped.norm(dim=0)            # shape [768]

            # -------------------------------------------------------------------------
        # 3. (Optional) Initialize or check quantizer
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        X = self.inp1.squeeze(0)
        W = W.to(X.dtype)
        with torch.enable_grad():
            print(X.shape,W.shape)
            mask = self.iterative_input_prune_STE(X, W, sparsity, prune_rows,prune_cols, num_iter=100)

        # 应用到 X 上:
        pruned_W = W * mask.unsqueeze(0)

        pruned_W = W.clone()
        # -----------------------------------------------

        # -------------------------------------------------------------------------
        # 5. (Optional) Quantize the pruned matrix
        # -------------------------------------------------------------------------
        if hasattr(self, 'quantizer'):
            # Some quantizers may require a certain shape
            pruned_W_q = pruned_W.unsqueeze(1)
            pruned_W_q = quantize(pruned_W_q, self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)
            pruned_W_q = pruned_W_q.squeeze(1)
        else:
            pruned_W_q = pruned_W

        # -------------------------------------------------------------------------
        # 6. Reshape back to original shape and assign to layer weights
        # -------------------------------------------------------------------------
        if isinstance(self.layer, nn.Conv2d):
            pruned_W_q = pruned_W_q.view(original_shape)
        elif hasattr(self.layer, '__class__') and self.layer.__class__.__name__ == 'Conv1D':
            # Remember we transposed for Conv1D
            pruned_W_q = pruned_W_q.t().view(original_shape)
        elif len(original_shape) > 2:
            pruned_W_q = pruned_W_q.view(original_shape)

        self.layer.weight.data = pruned_W_q.to(self.layer.weight.data.dtype)
        
        
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()