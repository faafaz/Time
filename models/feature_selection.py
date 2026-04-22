"""
Feature Selection Modules for Time Series Forecasting

Implements various learnable feature selection mechanisms:
1. GFLU (Gated Feature Learning Unit) - from GANDALF (2024)
2. Channel Attention - from CBAM/SENet
3. Variable Selection Network - from Temporal Fusion Transformer

All modules support ablation via enable/disable flags.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeatureSelection(nn.Module):
    """
    GFLU-style Gated Feature Selection with t-softmax for sparsity.
    
    Based on GANDALF (2024): https://arxiv.org/abs/2207.08548
    
    Key Features:
    - Learnable sparse feature masks via t-softmax
    - Minimal parameters (~n_features)
    - Controllable sparsity via t parameter
    - Optional gating mechanism for feature refinement
    
    Args:
        n_features: Number of input features
        sparsity_init: Initial sparsity parameter t (default: 0.5)
        learnable_t: Whether t is learnable (default: True)
        use_gating: Whether to use GRU-style gating (default: False, for simplicity)
        hidden_size: Hidden size for gating (only used if use_gating=True)
    
    Input:
        x: [B, T, C] - Batch, Time, Channels
    
    Output:
        out: [B, T, C] - Feature-selected output
        mask: [C] - Feature importance mask (for interpretability)
    """
    def __init__(self, n_features, sparsity_init=0.5, learnable_t=True, 
                 use_gating=False, hidden_size=None):
        super().__init__()
        self.n_features = n_features
        self.use_gating = use_gating
        
        # Feature mask parameters (one per feature)
        self.feature_params = nn.Parameter(torch.randn(n_features))
        
        # t-softmax sparsity parameter
        if learnable_t:
            self.t = nn.Parameter(torch.tensor(sparsity_init))
        else:
            self.register_buffer('t', torch.tensor(sparsity_init))
        
        # Optional gating mechanism (GRU-style)
        if use_gating:
            if hidden_size is None:
                hidden_size = n_features
            self.hidden_size = hidden_size
            
            # Update gate
            self.W_z = nn.Linear(n_features, hidden_size)
            self.U_z = nn.Linear(hidden_size, hidden_size)
            
            # Reset gate
            self.W_r = nn.Linear(n_features, hidden_size)
            self.U_r = nn.Linear(hidden_size, hidden_size)
            
            # Candidate hidden state
            self.W_h = nn.Linear(n_features, hidden_size)
            self.U_h = nn.Linear(hidden_size, hidden_size)
            
            # Output projection
            self.output_proj = nn.Linear(hidden_size, n_features)
    
    def t_softmax(self, x, t):
        """
        t-softmax: Sparse softmax activation.
        
        Formula:
            w_i = ReLU(x_i + t - max(x))
            t-softmax(x)_i = w_i * exp(x_i) / Σ(w_j * exp(x_j))
        
        Args:
            x: Input logits [C]
            t: Sparsity parameter (higher t = more sparse)
        
        Returns:
            Sparse probability distribution [C]
        """
        max_val = x.max()
        w = torch.relu(x + t - max_val)  # Sparsity weights
        numerator = w * torch.exp(x)
        denominator = (w * torch.exp(x)).sum() + 1e-8
        return numerator / denominator
    
    def forward(self, x, prev_hidden=None):
        """
        Forward pass with feature selection.
        
        Args:
            x: [B, T, C]
            prev_hidden: [B, H] - Previous hidden state (only for gating)
        
        Returns:
            out: [B, T, C]
            mask: [C] - Feature importance weights
            hidden: [B, H] - Current hidden state (only for gating)
        """
        B, T, C = x.shape
        
        # Compute feature selection mask
        mask = self.t_softmax(self.feature_params, self.t)  # [C]
        
        # Apply mask
        x_masked = x * mask.unsqueeze(0).unsqueeze(0)  # [B, T, C]
        
        if not self.use_gating:
            return x_masked, mask, None
        
        # GRU-style gating for feature refinement
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.hidden_size, device=x.device)
        
        # Average over time for gating
        x_avg = x_masked.mean(dim=1)  # [B, C]
        
        # Update gate
        z = torch.sigmoid(self.W_z(x_avg) + self.U_z(prev_hidden))  # [B, H]
        
        # Reset gate
        r = torch.sigmoid(self.W_r(x_avg) + self.U_r(prev_hidden))  # [B, H]
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x_avg) + self.U_h(r * prev_hidden))  # [B, H]
        
        # Update hidden state
        h_new = (1 - z) * prev_hidden + z * h_tilde  # [B, H]
        
        # Project back to feature space
        out_proj = self.output_proj(h_new)  # [B, C]
        
        # Combine with masked input
        out = x_masked + out_proj.unsqueeze(1)  # [B, T, C]
        
        return out, mask, h_new


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CBAM-style).
    
    Based on CBAM (2018) and SENet (2017).
    
    Key Features:
    - Global average and max pooling
    - Shared MLP for channel weighting
    - Minimal parameters (~C²/reduction)
    - Fast and effective
    
    Args:
        n_channels: Number of input channels
        reduction: Reduction ratio for MLP (default: 4)
    
    Input:
        x: [B, T, C]
    
    Output:
        out: [B, T, C]
        weight: [B, C] - Channel importance weights
    """
    def __init__(self, n_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Shared MLP
        hidden_dim = max(n_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [B, T, C]

        Returns:
            out: [B, T, C]
            weight: [B, C]
            hidden: None (for compatibility with GFLU interface)
        """
        B, T, C = x.shape

        # Global pooling
        x_t = x.transpose(1, 2)  # [B, C, T]
        avg_out = self.avg_pool(x_t).squeeze(-1)  # [B, C]
        max_out = self.max_pool(x_t).squeeze(-1)  # [B, C]

        # Channel attention weights
        avg_weight = self.mlp(avg_out)  # [B, C]
        max_weight = self.mlp(max_out)  # [B, C]
        weight = self.sigmoid(avg_weight + max_weight)  # [B, C]

        # Apply weights
        out = x * weight.unsqueeze(1)  # [B, T, C]

        return out, weight, None


class GRN(nn.Module):
    """
    Gated Residual Network (from Temporal Fusion Transformer).
    
    Key Features:
    - Gated linear unit (GLU) for feature transformation
    - Residual connection with skip projection
    - Layer normalization
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden dimension
        output_size: Output dimension
        dropout: Dropout probability
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Skip connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = nn.Identity()
        
        self.layer_norm = nn.LayerNorm(output_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [*, input_size]
        
        Returns:
            out: [*, output_size]
        """
        # Main path
        h = self.fc1(x)
        h = self.elu(h)
        h = self.dropout(h)
        
        # Gating
        gate = self.sigmoid(self.gate(h))
        h = self.fc2(h)
        h = gate * h
        
        # Residual + normalization
        out = self.layer_norm(h + self.skip(x))
        return out


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (from Temporal Fusion Transformer).
    
    Based on TFT (2019): https://arxiv.org/abs/1912.09363
    
    Key Features:
    - Per-variable GRN encoding
    - Softmax-based variable importance weights
    - Strong interpretability
    
    Args:
        n_vars: Number of input variables
        hidden_size: Hidden dimension for GRN
        dropout: Dropout probability
    
    Input:
        x: [B, T, C]
    
    Output:
        out: [B, T, C]
        weights: [B, C] - Time-averaged variable importance weights
    """
    def __init__(self, n_vars, hidden_size, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        
        # Per-variable GRN
        self.var_grns = nn.ModuleList([
            GRN(1, hidden_size, hidden_size, dropout)
            for _ in range(n_vars)
        ])
        
        # Variable selection weights
        self.var_selection = GRN(
            n_vars * hidden_size, 
            hidden_size, 
            n_vars, 
            dropout
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [B, T, C]
        
        Returns:
            out: [B, T, C]
            weights: [B, C] - Time-averaged variable importance weights
            hidden: None (for compatibility with GFLU interface)
        """
        B, T, C = x.shape

        # Per-variable encoding
        var_encodings = []
        for i in range(C):
            var_i = x[:, :, i:i+1]  # [B, T, 1]
            var_i_flat = var_i.reshape(B * T, 1)
            encoded = self.var_grns[i](var_i_flat)  # [B*T, hidden]
            var_encodings.append(encoded)

        # Concatenate all variable encodings
        all_vars = torch.cat(var_encodings, dim=-1)  # [B*T, C*hidden]

        # Variable selection weights
        weights = self.var_selection(all_vars)  # [B*T, C]
        weights = self.softmax(weights)  # [B*T, C]
        weights = weights.reshape(B, T, C)  # [B, T, C]

        # Weighted output
        out = x * weights

        # Return time-averaged weights for interpretability
        return out, weights.mean(dim=1), None  # [B, C], None

