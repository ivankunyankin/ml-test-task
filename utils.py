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


def custom_collate(data):

    """
   data: is a list of tuples with (melspec, transcript, input_length, label_length), where:
    - 'melspec' is a tensor of arbitrary shape
    - 'transcript' is an encoded transcript - list of integers
    - input_length - is length of the spectrogram - represents time - int
    - label_length - is length of the encoded label - int
    """

    melspecs, texts, input_lengths, label_lengths = zip(*data)

    specs = [torch.transpose(spec, 0, 1) for spec in melspecs]
    specs = pad_sequence(specs, batch_first=True)
    specs = torch.transpose(specs, 1, 2)

    labels = pad_sequence(texts, batch_first=True)

    return specs, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)


def decode(output, labels, label_lengths, blank_label=10, collapse_repeated=True):

    arg_maxes = torch.argmax(output, dim=2)

    decodes = []
    targets = []

    for i, args in enumerate(arg_maxes):  # for each sample in the batch
        decode = []

        ints = labels[i][:label_lengths[i]].tolist()
        targets.append("".join([str(i) for i in ints]))

        for j, index in enumerate(args):  # for each predicted character in the sample
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(str(index.item()))

        decodes.append("".join(decode))

    return decodes, targets


def cer(ground_truth, hypothesis):

    ground_truth = [char for seq in ground_truth for char in seq]
    hypothesis = [char for seq in hypothesis for char in seq]

    return wer(ground_truth, hypothesis)