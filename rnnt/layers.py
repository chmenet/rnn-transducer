import torch
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression, dynamic_range_decompression, mel_normalize, mel_denormalize
from stft import STFT

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, hparams):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.sampling_rate = hparams.sampling_rate
        self.stft_fn = STFT(hparams.filter_length, hparams.hop_length, hparams.win_length)
        self.max_abs_mel_value = hparams.max_abs_mel_value
        mel_basis = librosa_mel_fn(
            hparams.sampling_rate, hparams.filter_length, hparams.n_mel_channels, hparams.mel_fmin, hparams.mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def transform2(self, y):
        result = torch.stft(y, n_fft=1024, hop_length=256, win_length =1024)
        print(result.shape)
        real = result[:, :, :, 0]
        imag = result[:, :, :, 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.autograd.Variable(
            torch.atan2(imag.data, real.data))
        print(phase.shape, phase.min(), phase.max())
        print(magnitude.shape, magnitude.min(), magnitude.max())
        return magnitude, phase

    def cepstrum_from_mel(self, mel, ref_level_db = 20, magnitude_power=1.5):
        assert (torch.min(mel.data) >= -self.max_abs_mel_value)
        assert (torch.max(mel.data) <= self.max_abs_mel_value)
        #print('mel: ', mel.max(), mel.min())
        spec = mel_denormalize(mel, self.max_abs_mel_value)
        #print('spec: ', spec.max(), spec.min())
        magnitudes = self.spectral_de_normalize(spec + ref_level_db) ** (1 / magnitude_power)
        #print('Magnitude: ', Magnitude.max(), Magnitude.min())
        pow_spec = (magnitudes**2)/1024 # if filter_length = 1024
        #print('pow_spec: ', pow_spec.max(), pow_spec.min())
        db_pow_spec = torch.log(torch.clamp(pow_spec,min=1e-5))*20 #db
        #print('db_pow_spec: ', db_pow_spec.max(), db_pow_spec.min())
        mcc = dct(db_pow_spec,'ortho' )
        return mcc

    def cepstrum_from_audio(self, y):
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        #print('magnitudes: ', magnitudes.max(), magnitudes.min())
        pow_spec = (1/1024)*magnitudes**2
        #print('pow_spec: ', pow_spec.max(), pow_spec.min())
        mel_spectrogram = torch.matmul(self.mel_basis, pow_spec).squeeze(0).transpose(0,1)
        #print('mel_spectrogram: ', mel_spectrogram.max(), mel_spectrogram.min())
        db_mel_spectrogram = torch.log10(torch.clamp(pow_spec,min=1e-5))*20 #db
        #print('db_mel_spectrogram: ', db_mel_spectrogram.max(), db_mel_spectrogram.min())
        mcc = dct(db_mel_spectrogram,'ortho')
        return mcc

    def mel_spectrogram(self, y, ref_level_db = 20, magnitude_power=1.5):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        #print('y' ,y.max(), y.mean(), y.min())
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        #print('stft_fn', magnitudes.max(), magnitudes.mean(), magnitudes.min())
        mel_output = torch.matmul(self.mel_basis, torch.abs(magnitudes)**magnitude_power)
        #print('_linear_to_mel', mel_output.max(), mel_output.mean(), mel_output.min())
        mel_output = self.spectral_normalize(mel_output) - ref_level_db
        #print('_amp_to_db', mel_output.max(), mel_output.mean(), mel_output.min())
        mel_output = mel_normalize(mel_output, self.max_abs_mel_value)
        #print('_normalize', mel_output.max(), mel_output.mean(), mel_output.min())
        #spec = mel_denormalize(mel_output, self.max_abs_mel_value)
        #print('_denormalize', spec.max(), spec.mean(), spec.min())
        #spec = self.spectral_de_normalize(spec + ref_level_db)**(1/magnitude_power)
        #print('db_to_amp', spec.max(), spec.mean(), spec.min())
        return mel_output
