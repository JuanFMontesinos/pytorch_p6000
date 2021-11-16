import torch
from VnBSS.models.llcp_net import LlcpNet

device = torch.device('cuda:0')

# AUDIO AND VIDEO PARAMETERS
AUDIO_LENGTH = (4 * 2 ** 14 - 1)
N_VIDEO_FRAMES = 100
N_SKELETON_FRAMES = 100

AUDIO_SAMPLERATE = 16384
VIDEO_FRAMERATE = 25
AUDIO_VIDEO_RATE = AUDIO_SAMPLERATE / VIDEO_FRAMERATE

N_FFT = 1022
HOP_LENGTH = 256
N_MEL = 80
SP_FREQ_SHAPE = N_FFT // 2 + 1

MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]
DEBUG = {'isnan': True, 'ds_autogen': False, "overfit": False, 'verbose': False}

model = LlcpNet(debug=DEBUG,
                loss_criterion='MSE',
                remix_input=True,
                remix_coef=0.5,
                audio_length=AUDIO_LENGTH,
                audio_samplerate=AUDIO_SAMPLERATE,
                mean=MEAN,
                std=STD,
                n_fft=N_FFT, hop_length=HOP_LENGTH, n_mel=N_MEL,
                sp_freq_shape=SP_FREQ_SHAPE,
                log_sp_enabled=False,
                mel_enabled=False,
                complex_enabled=True,
                weighted_loss=True, loss_on_mask=True, binary_mask=False,
                downsample_coarse=True, downsample_interp=False,
                video_enabled=False,
                skeleton_enabled=False,
                single_frame_enabled=False,
                single_emb_enabled=False,
                llcp_enabled=True,
                device=device)

model = model.to(device)
inputs = {
    'llcp_embedding': torch.rand(2, 100, 512, device=device),
    'audio': torch.rand(2, AUDIO_LENGTH, device=device),
    'audio_acmt': torch.rand(2, AUDIO_LENGTH, device=device),
}
#         - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[
    ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
