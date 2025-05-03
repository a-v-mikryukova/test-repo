import torch
import torch.nn as nn
import torchaudio

from .dataset import LibriSpeechDataset

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
         1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')


def get_featurizer(sample_rate, n_mels, train=True):
    if train:
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels),
            torchaudio.transforms.FrequencyMasking(30),
            torchaudio.transforms.TimeMasking(100)
        )
    return torchaudio.transforms.MelSpectrogram()


def collate_fn(batch, text_transform, sample_rate=16000, n_mels=128, feature_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    audio_transform = get_featurizer(sample_rate, n_mels, train=(feature_type == "train"))

    for (waveform, _, utterance, _, _, _) in batch:
        spec = audio_transform(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)

        label = torch.tensor(text_transform.text_to_int(utterance.lower()), dtype=torch.long)
        labels.append(label)

        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(
        spectrograms,
        batch_first=True
    ).unsqueeze(1).transpose(2, 3)

    labels = nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True
    )

    return (
        spectrograms,
        labels,
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(label_lengths, dtype=torch.long)
    )