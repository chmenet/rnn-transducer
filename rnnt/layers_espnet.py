import torch
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from rnnt.audio_processing import dynamic_range_compression, dynamic_range_decompression, mel_normalize, mel_denormalize
from rnnt.stft_espnet import STFT

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

def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    m = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)
    return m

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
    def __init__(self, config):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = config.n_mel_channels
        self.sampling_rate = config.sampling_rate
        self.stft_fn = STFT(config.filter_length, config.hop_length, config.win_length)
        self.max_abs_mel_value = config.max_abs_mel_value
        mel_basis = librosa_mel_fn(
            config.sampling_rate, config.filter_length, config.n_mel_channels, config.mel_fmin, config.mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        bias = [1.0889494, 1.2151777, 0.9595682, 0.5710948, 0.11525655, \
                0.1650035, 0.4663777, 0.67052126, 0.7382921, 0.64285976, \
                0.66845304, 0.8032008, 0.88970685, 0.9809479, 1.0312854, \
                1.115497, 1.3010043, 1.3612534, 1.4133714, 1.4442914, \
                1.4157497, 1.5602378, 1.6092354, 1.6732794, 1.7499422, \
                1.7426846, 1.8945292, 1.977941, 2.0066006, 2.0312035, \
                2.205933, 2.129342, 2.2962654, 2.2588577, 2.3295689, \
                2.3560326, 2.3770041, 2.4039962, 2.4370804, 2.4806125, \
                2.5315294, 2.5859036, 2.642157, 2.6723998, 2.7151663, \
                2.7272666, 2.7331805, 2.704286, 2.685495, 2.6736267, \
                2.6747813, 2.6920958, 2.7349966, 2.756284, 2.7847264, \
                2.7919326, 2.8164277, 2.8354008, 2.8466384, 2.8630826, \
                2.8692098, 2.8939328, 2.938372, 3.01084, 3.1018033, \
                3.19208, 3.2841125, 3.3641038, 3.4380672, 3.502122, \
                3.5573611, 3.5981922, 3.6273797, 3.6428063, 3.6461506, \
                3.6408007, 3.6301754, 3.617861, 3.6390467, 3.7570682]
        scale = [0.88847476, 0.6545963, 0.5477136, 0.5062988, 0.46672204, \
                 0.46109515, 0.4793967, 0.49452525, 0.48896265, 0.47259146, \
                 0.46541813, 0.46840945, 0.47576383, 0.48573217, 0.49419498, \
                 0.49950126, 0.5078051, 0.51368976, 0.5210506, 0.5287385, \
                 0.5334307, 0.54625326, 0.55273384, 0.5596952, 0.5672863, \
                 0.56911933, 0.5822014, 0.59008205, 0.5935132, 0.59585124, \
                 0.6151175, 0.6086165, 0.6330749, 0.6317565, 0.6464341, \
                 0.65609115, 0.6625604, 0.6697218, 0.6787185, 0.6899366, \
                 0.7031865, 0.71767193, 0.73331404, 0.7443723, 0.7588258, \
                 0.7657399, 0.7698718, 0.7632448, 0.75874346, 0.7555333, \
                 0.75495964, 0.7579142, 0.7677287, 0.7729095, 0.7809229, \
                 0.7834592, 0.79204774, 0.7986897, 0.8021473, 0.80742586, \
                 0.8089727, 0.8151319, 0.82667005, 0.8496202, 0.8861803, \
                 0.93025696, 0.98163086, 1.031578, 1.0862143, 1.1438614, \
                 1.204122, 1.2553384, 1.2967699, 1.3214465, 1.3288966, \
                 1.3230902, 1.3084877, 1.2917982, 1.3366529, 1.6547843]
        self.bias = torch.FloatTensor(bias)
        self.scale = torch.FloatTensor(scale)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def transform2(self, y):
        result = torch.stft(y, n_fft=1024, hop_length=256, win_length =1024)
        #print(result.shape)
        real = result[:, :, :, 0]
        imag = result[:, :, :, 1]
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.autograd.Variable(
            torch.atan2(imag.data, real.data))
        #print(phase.shape, phase.min(), phase.max())
        #print(magnitude.shape, magnitude.min(), magnitude.max())
        return magnitude, phase

    def cepstrum_from_mel(self, mel, ref_level_db = 20, magnitude_power=1.5):
        assert (torch.min(mel.data) >= -self.max_abs_mel_value)
        assert (torch.max(mel.data) <= self.max_abs_mel_value)
        #print('mel: ', mel.max(), mel.min())
        spec = mel_denormalize(mel, self.max_abs_mel_value)
        #print('spec: ', spec.max(), spec.min())
        magnitudes = self.spectral_de_normalize(spec + ref_level_db).pow_(1 / magnitude_power)
        #print('Magnitude: ', Magnitude.max(), Magnitude.min())
        pow_spec = (magnitudes.pow_(2).mul_(1/1024)) # if filter_length = 1024
        #print('pow_spec: ', pow_spec.max(), pow_spec.min())
        db_pow_spec = pow_spec.clamp_(min=1e-5).log_().mul_(20) #db
        #print('db_pow_spec: ', db_pow_spec.max(), db_pow_spec.min())
        mcc = dct(db_pow_spec,'ortho' )
        return mcc

    def cepstrum_from_audio(self, y):
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)
        magnitudes, phases = self.stft_fn.transform(y)
        #print('magnitudes: ', magnitudes.max(), magnitudes.min())
        pow_spec = magnitudes.pow_(2).mul_(1/1024)
        #print('pow_spec: ', pow_spec.max(), pow_spec.min())
        mel_spectrogram = torch.matmul(self.mel_basis, pow_spec).squeeze_(0).transpose_(0,1)
        #print('mel_spectrogram: ', mel_spectrogram.max(), mel_spectrogram.min())
        db_mel_spectrogram = pow_spec.clamp_(min=1e-5).log10_().mul_(20) #db
        #print('db_mel_spectrogram: ', db_mel_spectrogram.max(), db_mel_spectrogram.min())
        mcc = dct(db_mel_spectrogram,'ortho')
        return mcc

    def mel_spectrogram(self, y, ref_level_db = 20, magnitude_power=1.5, isDebugging = False):
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

        if(isDebugging): print('y' ,y.max(), y.mean(), y.min())
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        if(isDebugging): print('stft_fn', magnitudes.max(), magnitudes.mean(), magnitudes.min())
        #mel_output = torch.matmul(self.mel_basis, torch.abs(magnitudes)**magnitude_power)

        # power magnitude to mel_basis
        mel_output = torch.matmul(self.mel_basis, magnitudes.abs_().pow_(magnitude_power))
        if(isDebugging): print('_linear_to_mel', mel_output.max(), mel_output.mean(), mel_output.min())

        # mel_basis to db with clipping, - ref_level_dbz
        mel_output = self.spectral_normalize(mel_output).add_(- ref_level_db)
        if(isDebugging): print('_amp_to_db', mel_output.max(), mel_output.mean(), mel_output.min())

        # normalized any scale to [-4,4]
        mel_output = mel_normalize(mel_output, self.max_abs_mel_value)
        if(isDebugging): print('_normalize', mel_output.max(), mel_output.mean(), mel_output.min())

        #spec = mel_denormalize(mel_output)
        #print('_denormalize', spec.max(), spec.mean(), spec.min())
        #spec = self.spectral_de_normalize(spec + ref_level_db)**(1/magnitude_power)
        #print('db_to_amp', spec.max(), spec.mean(), spec.min())
        return mel_output

    # for torch script
    def forward(self, y: torch.Tensor):
        # torch script는 self.var \in 기본 변수 {float, int, ..}의 참조를 허용하지 않음.
        ref_level_db = 20.0
        magnitude_power = 1.5
        C = 20
        clip_val = 1e-5
        max_abs_value = 4.0
        min_level_db = -100
        max_len_featre = 300

        left_context_width = 0
        right_context_width = 0

        frame_rate = 10

        # melbasis and power spectrogram
        magnitudes = self.stft_fn.forward(y)

        mel_output = torch.matmul(self.mel_basis, magnitudes.abs_().pow_(magnitude_power))

        # clip and db scale
        mel_output.clamp_(min=clip_val).log10_().mul_(C).add_(- ref_level_db)

        # normalized mel
        mel_output.add_(-min_level_db).mul_(2 * max_abs_value).mul_(-1 / min_level_db).add_(-max_abs_value).clamp_(min=-max_abs_value, max=max_abs_value)

        mel_output.transpose_(1, 2).squeeze_(0)

        features = mel_output

        # concat frames
        time_steps, features_dim = features.shape

        concated_features = torch.zeros(
            time_steps, features_dim *
                   (1 + left_context_width + right_context_width),
            dtype=torch.float32)
        # middle part is just the uttarnce
        concated_features[:, left_context_width * features_dim:
                             (left_context_width + 1) * features_dim] = features

        for i in range(left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
            (left_context_width - i - 1) * features_dim:
            (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
            (right_context_width + i + 1) * features_dim:
            (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        # subsampled features
        features = concated_features

        interval = int(frame_rate / 10)
        temp_mat = [features[i]
                    for i in range(0, features.shape[0], interval)]
        subsampled_features = torch.stack(temp_mat)

        # last shape
        time_steps, features_dim = subsampled_features.shape
        tend = time_steps if time_steps < max_len_featre else max_len_featre
        last_feature = torch.ones(max_len_featre, features_dim, dtype=torch.float32)*-4.0
        last_feature[:tend, :] = subsampled_features[:tend, :]
        last_feature.add_(self.bias[None,:]).mul_(self.scale[None,:])
        subsampled_features.add_(self.bias[None, :]).mul_(self.scale[None, :])

        return subsampled_features