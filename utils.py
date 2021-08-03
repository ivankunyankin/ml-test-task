import torch
import librosa
import numpy as np
import matplotlib as matplotlib
import matplotlib.cm
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence


def audio_to_mel(x, hparams):

    spec = librosa.feature.melspectrogram(
        x,
        sr=hparams["sr"],
        n_fft=hparams["n_fft"],
        win_length=hparams["win_length"],
        hop_length=hparams["hop_length"],
        power=1,
        fmin=0,
        fmax=8000,
        n_mels=hparams["n_mels"]
    )

    spec = np.log(np.clip(spec, a_min=1e-5, a_max=None))
    spec = torch.FloatTensor(spec)

    return spec


def save_spec(spec):

    cm = matplotlib.cm.get_cmap('gray')

    normed = (spec - spec.min()) / (spec.max() - spec.min())
    mapped = cm(normed)

    return torch.from_numpy(mapped).flip(0).permute(2, 0, 1)


def cer(ground_truth, hypothesis):

    ground_truth = [char for seq in ground_truth for char in seq]
    hypothesis = [char for seq in hypothesis for char in seq]

    return wer(ground_truth, hypothesis)


def decode(outputs, labels):

    args = torch.argmax(outputs, dim=1)

    decoded_preds = []
    for item in args:
        pred = "".join(str(int(i)) for i in item)
        decoded_preds.append(pred)

    decoded_targets = []
    for item in labels:
        target = "".join(str(int(i)) for i in item)
        decoded_targets.append(target)

    return decoded_preds, decoded_targets
