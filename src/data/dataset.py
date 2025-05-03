import torchaudio
from torch.utils.data import Dataset
import os


class LibriSpeechDataset(Dataset):
    def __init__(self, data_dir, url, download=True):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=data_dir,
            url=url,
            download=download,
            folder_in_archive=os.path.join(data_dir, "LibriSpeech")
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)