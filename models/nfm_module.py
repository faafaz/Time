"""
Neural Fourier Modelling (NFM) Module for Target Variable Prediction

This is a lightweight implementation of NFM specifically designed for:
1. Processing the original target variable from raw input data
2. Frequency domain modeling using DFT/IDFT
3. Minimal parameter overhead
4. Easy integration with iTransformer_xLSTM

Reference: Neural Fourier Modelling (NFM) - A Highly Compact Approach to Time-Series Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Vanilla Transformer"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, batch_size):
        # Return [B, L, d_model]
        return self.pe.unsqueeze(0).expand(batch_size, -1, -1)


class SirenLayer(nn.Module):
    """SIREN layer with sinusoidal activation"""
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                           1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / self.linear.in_features) / self.omega,
                                           math.sqrt(6 / self.linear.in_features) / self.omega)
    
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SirenBlock(nn.Module):
    """SIREN network for generating frequency tokens or filters"""
    def __init__(self, dim_in, hidden_dim, out_dim, omega=30.0, num_layers=2):
        super().__init__()
        layers = []
        layers.append(SirenLayer(dim_in, hidden_dim, omega=omega, is_first=True))
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class LFTLayer(nn.Module):
    """
    Learnable Frequency Token (LFT) Layer
    Learns effective spectral priors and enables frequency extension
    """
    def __init__(self, hidden_dim, seq_len, siren_hidden=32, omega=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.freq_dim = seq_len // 2 + 1  # rfft output dimension
        
        # SIREN network to generate frequency tokens
        self.freq_token_generator = SirenBlock(
            dim_in=1,  # Frequency index as input
            hidden_dim=siren_hidden,
            out_dim=hidden_dim,
            omega=omega,
            num_layers=2
        )
        
        # Learnable scale and bias for frequency tokens
        self.scale = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, C] time domain input
        Returns:
            x_base: [B, L, C] processed time domain output
        """
        B, L, C = x.shape
        
        # Transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, F, C] complex
        
        # Generate frequency tokens
        freq_indices = torch.linspace(0, 1, self.freq_dim, device=x.device).view(-1, 1)  # [F, 1]
        freq_tokens = self.freq_token_generator(freq_indices)  # [F, hidden_dim]
        freq_tokens = freq_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, F, hidden_dim]
        
        # Apply learnable scale and bias
        freq_tokens = self.scale * freq_tokens + self.bias
        
        # Map frequency representation (simplified version)
        # In full NFM, this involves complex frequency mapping
        # Here we use a simplified approach: element-wise multiplication in frequency domain
        if C == self.hidden_dim:
            # Apply frequency tokens as modulation
            x_freq_real = x_freq.real * torch.cos(freq_tokens.mean(dim=-1, keepdim=True))
            x_freq_imag = x_freq.imag * torch.sin(freq_tokens.mean(dim=-1, keepdim=True))
            x_freq_modulated = torch.complex(x_freq_real, x_freq_imag)
        else:
            x_freq_modulated = x_freq
        
        # Transform back to time domain
        x_base = torch.fft.irfft(x_freq_modulated, n=L, dim=1, norm='ortho')  # [B, L, C]
        
        return self.norm(x_base)


class INFFLayer(nn.Module):
    """
    Implicit Neural Fourier Filter (INFF) Layer
    Instance-adaptive and mode-aware frequency filtering
    """
    def __init__(self, hidden_dim, siren_hidden=32, omega=30.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # SIREN network to generate filter
        self.filter_generator = SirenBlock(
            dim_in=1,  # Frequency index as input
            hidden_dim=siren_hidden,
            out_dim=hidden_dim,
            omega=omega,
            num_layers=2
        )
        
        # Learnable scale and bias for filter
        self.scale = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Complex-valued MLP (simplified: process real and imaginary separately)
        self.cv_mlp_real = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cv_mlp_imag = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, conditional=None):
        """
        Args:
            x: [B, L, C] time domain input
            conditional: [B, L, C] conditional information (optional)
        Returns:
            x_filtered: [B, L, C] filtered time domain output
        """
        B, L, C = x.shape
        F = L // 2 + 1
        
        # Transform to frequency domain
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, F, C] complex
        
        # Generate filter using SIREN
        freq_indices = torch.linspace(0, 1, F, device=x.device).view(-1, 1)  # [F, 1]
        filter_weights = self.filter_generator(freq_indices)  # [F, hidden_dim]
        filter_weights = filter_weights.unsqueeze(0).expand(B, -1, -1)  # [B, F, hidden_dim]
        
        # Apply scale and bias
        filter_weights = self.scale * filter_weights + self.bias
        
        # Add conditional information if provided
        if conditional is not None:
            cond_freq = torch.fft.rfft(conditional, dim=1, norm='ortho')
            filter_weights = filter_weights + cond_freq.real
        
        # Apply complex-valued MLP
        if C == self.hidden_dim:
            filter_real = self.cv_mlp_real(filter_weights)
            filter_imag = self.cv_mlp_imag(filter_weights)
            filter_complex = torch.complex(filter_real, filter_imag)
            
            # Apply filter in frequency domain
            x_freq_filtered = x_freq * filter_complex
        else:
            x_freq_filtered = x_freq
        
        # Transform back to time domain
        x_filtered = torch.fft.irfft(x_freq_filtered, n=L, dim=1, norm='ortho')
        
        return self.norm(x_filtered)


class MixerBlock(nn.Module):
    """
    Mixer block with channel mixing and token mixing (in frequency domain)
    """
    def __init__(self, hidden_dim, hidden_factor=3, dropout=0.1, siren_hidden=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Channel mixing (position-wise FFN)
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * hidden_factor),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * hidden_factor, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Token mixing (INFF in frequency domain)
        self.token_mixer = INFFLayer(hidden_dim, siren_hidden=siren_hidden)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, conditional=None):
        """
        Args:
            x: [B, L, C]
            conditional: [B, L, C] (optional)
        Returns:
            x: [B, L, C]
        """
        # Channel mixing
        x = x + self.channel_mixer(self.norm1(x))
        
        # Token mixing (frequency domain)
        x = x + self.token_mixer(self.norm2(x), conditional)
        
        return x


class NFMPredictor(nn.Module):
    """
    Lightweight NFM predictor for target variable forecasting
    
    This module:
    1. Takes raw target variable as input [B, seq_len, 1]
    2. Projects to hidden dimension
    3. Applies LFT for frequency token learning
    4. Applies Mixer blocks for frequency domain processing
    5. Projects to prediction length
    """
    def __init__(self, seq_len, pred_len, hidden_dim=32, num_layers=1, 
                 hidden_factor=3, dropout=0.1, siren_hidden=32, omega=30.0):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # Dual projection paths (from NFM paper)
        self.projection_in1 = nn.Linear(1, hidden_dim, bias=False)
        self.projection_in2 = nn.Sequential(
            nn.Linear(1, hidden_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        )
        
        # Positional encoding
        self.pos_emb = SinusoidalPositionalEncoding(hidden_dim, max_len=seq_len)
        
        # LFT layer
        self.lft_layer = LFTLayer(hidden_dim, seq_len, siren_hidden=siren_hidden, omega=omega)
        
        # Mixer layers
        self.mixer_layers = nn.ModuleList([
            MixerBlock(hidden_dim, hidden_factor=hidden_factor, dropout=dropout, siren_hidden=siren_hidden)
            for _ in range(num_layers)
        ])
        
        # Final FFN
        self.final_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Forecasting head
        self.forecaster = nn.Linear(hidden_dim, 1, bias=True)
        
        # Time projection (seq_len -> pred_len)
        self.time_proj = nn.Linear(seq_len, pred_len)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in [layer for layer in self.lft_layer.modules()] \
                                        and m not in [layer for mixer in self.mixer_layers for layer in mixer.modules()]:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, 1] - raw target variable
        Returns:
            pred: [B, pred_len, 1] - prediction
        """
        B, L, C = x.shape
        assert L == self.seq_len and C == 1, f"Expected input shape [B, {self.seq_len}, 1], got {x.shape}"
        
        # Instance Normalization (RevIN-style)
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, 1]
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # [B, 1, 1]
        x_norm = (x - x_mean) / x_std
        
        # Dual projection
        x1 = self.projection_in1(x_norm)  # [B, L, hidden_dim]
        x2 = self.projection_in2(x_norm)  # [B, L, hidden_dim]
        x_proj = x1 + x2  # [B, L, hidden_dim]
        
        # LFT layer (frequency token learning)
        z = self.lft_layer(x_proj)  # [B, L, hidden_dim]
        residual = z
        conditional = z
        
        # Add positional encoding
        z = z + self.pos_emb(B)[:, :L, :]  # [B, L, hidden_dim]
        
        # Mixer layers
        for mixer in self.mixer_layers:
            z = mixer(z, conditional=conditional)  # [B, L, hidden_dim]
        
        # Add residual and final FFN
        z = z + residual
        z = self.final_ffn(z)  # [B, L, hidden_dim]
        
        # Forecasting head
        z = self.forecaster(z)  # [B, L, 1]
        
        # Time projection: seq_len -> pred_len
        z = z.transpose(1, 2)  # [B, 1, L]
        z = self.time_proj(z)  # [B, 1, pred_len]
        z = z.transpose(1, 2)  # [B, pred_len, 1]
        
        # Denormalization
        pred = z * x_std + x_mean
        
        return pred


class LightweightFusion(nn.Module):
    """
    Lightweight fusion module for combining LSTM and NFM predictions
    
    Supports three fusion strategies:
    1. 'gate': Learnable gating mechanism
    2. 'add': Weighted addition with learnable weights
    3. 'concat': Concatenation followed by linear projection
    """
    def __init__(self, fusion_type='gate', hidden_dim=64):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'gate':
            # Gating mechanism: learns to weight two predictions
            self.gate = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'add':
            # Simple weighted addition
            self.weights = nn.Parameter(torch.tensor([0.7, 0.3]))
        elif fusion_type == 'concat':
            # Concatenation + linear
            self.proj = nn.Linear(2, 1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, lstm_pred, nfm_pred):
        """
        Args:
            lstm_pred: [B, pred_len, 1] - LSTM prediction
            nfm_pred: [B, pred_len, 1] - NFM prediction
        Returns:
            fused_pred: [B, pred_len, 1] - fused prediction
        """
        if self.fusion_type == 'gate':
            # Stack predictions
            stacked = torch.cat([lstm_pred, nfm_pred], dim=-1)  # [B, pred_len, 2]
            # Compute gates
            gates = self.gate(stacked)  # [B, pred_len, 2]
            # Weighted sum
            fused = gates[:, :, 0:1] * lstm_pred + gates[:, :, 1:2] * nfm_pred
        elif self.fusion_type == 'add':
            # Softmax weights
            w = F.softmax(self.weights, dim=0)
            fused = w[0] * lstm_pred + w[1] * nfm_pred
        elif self.fusion_type == 'concat':
            # Concatenate and project
            stacked = torch.cat([lstm_pred, nfm_pred], dim=-1)  # [B, pred_len, 2]
            fused = self.proj(stacked)  # [B, pred_len, 1]
        
        return fused

