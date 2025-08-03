import torch
import torch.nn as nn

# Reversible Instance Normalization
class RIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x, reverse=False):
        # x shape: (batch_size, channels, seq_len)
        if not reverse:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + self.eps
            self.mean = mean
            self.std = std
            return (x - mean) / std
        else:
            return x * self.std + self.mean

# Complex-Valued Linear Layer
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real_linear = nn.Linear(in_features, out_features)
        self.imag_linear = nn.Linear(in_features, out_features)
        
    def forward(self, real, imag):
        real_out = self.real_linear(real) - self.imag_linear(imag)
        imag_out = self.real_linear(imag) + self.imag_linear(real)
        return real_out, imag_out

# FITS Model
class FITS(nn.Module):
    def __init__(self, look_back, horizon, cof, channels):
        """
        Args:
            look_back: Input sequence length
            horizon: Forecast horizon
            cof: Cutoff frequency (number of frequency components to keep)
            channels: Number of input channels
        """
        super().__init__()

        if cof > look_back // 2 + 1:
            raise ValueError(f"Cutoff frequency (cof={cof}) cannot be greater than max frequency ({look_back//2+1}).")

        self.look_back = look_back
        self.horizon = horizon
        self.cof = cof
        self.channels = channels
        
        # Calculate output length after interpolation
        self.output_length = look_back + horizon
        self.freq_out_len = self.output_length // 2 + 1
        
        # Modules
        self.rin = RIN()
        self.complex_linear = ComplexLinear(cof, int(cof * self.output_length / look_back))
        
    def forward(self, x):
        # x shape: (batch_size, channels, look_back)
        batch_size = x.shape[0]
        
        x_norm = self.rin(x)
        
        # Real FFT
        x_freq = torch.fft.rfft(x_norm, dim=-1, norm='ortho')
        
        # Low-pass filter
        x_freq = x_freq[..., :self.cof]
        
        real = x_freq.real
        imag = x_freq.imag
        
        # Flatten batch and channel dimensions
        real = real.reshape(-1, self.cof)
        imag = imag.reshape(-1, self.cof)
        
        real_out, imag_out = self.complex_linear(real, imag)
        
        # Zero-padding
        pad_len = self.freq_out_len - real_out.shape[-1]
        real_out = nn.functional.pad(real_out, (0, pad_len))
        imag_out = nn.functional.pad(imag_out, (0, pad_len))
        

        complex_out = torch.complex(real_out, imag_out)
        complex_out = complex_out.reshape(batch_size, self.channels, -1)
        output = torch.fft.irfft(complex_out, n=self.output_length, dim=-1, norm='ortho')
        
        # Reverse normalization
        output = self.rin(output, reverse=True)
        
        return output 
