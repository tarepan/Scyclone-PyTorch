from typing import NamedTuple, Optional
import torch
from torch.tensor import Tensor
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule

# currently there is no stub in npvcc2016
from npvcc2016.PyTorch.dataset.spectrogram import NpVCC2016_spec  # type: ignore
from jsut.PyTorch.dataset.spectrogram import JSUT_spec
from jsss.PyTorch.dataset.spectrogram import JSSS_spec

class DataLoaderPerformance(NamedTuple):
    """
    All attributes which affect performance of [torch.utils.data.DataLoader][^DataLoader] @ v1.6.0
    [^DataLoader]:https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """

    num_workers: int
    pin_memory: bool


class NonParallelSpecDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        performance: DataLoaderPerformance = DataLoaderPerformance(4, True),
    ):
        super().__init__()
        self.batch_size = batch_size
        self._num_worker = performance.num_workers
        self._pin_memory = performance.pin_memory

    def prepare_data(self, *args, **kwargs) -> None:
        NonParallelSpecBigDataset(train=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_full = NonParallelSpecBigDataset(train=True)
            # use modulo for validation (#training become batch*N)
            n_full = len(dataset_full)
            mod = n_full % self.batch_size
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [n_full - mod, mod]
            )
            self.batch_size_val = mod
        if stage == "test" or stage is None:
            self.dataset_test = NonParallelSpecBigDataset(train=False)
            self.batch_size_test = self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size_val,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size_test,
            num_workers=self._num_worker,
            pin_memory=self._pin_memory,
        )


def pad_last_dim(d: Tensor, length_min: int = 160) -> Tensor:
    """
    Pad last dimension with 0 if length is not enough.
    If input is longer than `length_min`, nothing happens.
    [..., L<160] => [..., L==160]
    """
    shape = d.size()
    length_d = shape[-1]
    if length_d < length_min:
        a = torch.zeros([*shape[:-1], length_min - length_d])
        return torch.cat((d, a), -1)
    else:
        return d


def slice_last_dim(d: Tensor, length: int = 160) -> Tensor:
    """
    Slice last dimention if length is too much.
    If input is shorter than `length`, error is thrown.
    [..., L>160] => [..., L==160]
    """
    start = torch.randint(0, d.size()[-1] - (length - 1), (1,)).item()
    return torch.narrow(d, -1, start, length)


def pad_clip(d: Tensor) -> Tensor:
    return slice_last_dim(pad_last_dim(d))


class NonParallelSpecDataset(Dataset):
    def __init__(self, train: bool):
        self.datasetA = NpVCC2016_spec(train, ["SF1"], pad_clip, True)
        self.datasetB = NpVCC2016_spec(train, ["TF2"], pad_clip, True)

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.
        Potential problem: A/B pair
        Current implementation yield fixed A/B pair.
        When batch size is small (e.g. 1), Batch_A and Batch_B has strong correlation.
        If big batch, correlation decrease so little problem.
        We could solve this problem through sampler (e.g. sampler + sampler reset).
        """
        # ignore label
        return (self.datasetA[n][0], self.datasetB[n][0])

    def __len__(self) -> int:
        return min(len(self.datasetA), len(self.datasetB))


class NonParallelSpecBigDataset(Dataset):
    def __init__(self, train: bool):

        resampled_sr = 16000
        if train:
            subtypes_a = ["basic5000"]
            subtypes_b = ["short-form/basic5000"]
        else:
            subtypes_a = ["voiceactress100"]
            subtypes_b = ["short-form/voiceactress100"]

        self.datasetA = JSUT_spec(train, download_corpus=True, transform=pad_clip, subtypes=subtypes_a,
            # corpus_adress=,
            # dataset_adress=,
            resample_sr=resampled_sr
        )
        self.datasetB = JSSS_spec(train, download_corpus=True, transform=pad_clip, subtypes=subtypes_b,
            # corpus_adress=,
            # dataset_adress=,
            resample_sr=resampled_sr
        )

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Potential problem: A/B pair
        Current implementation yield fixed A/B pair.
        When batch size is small (e.g. 1), Batch_A and Batch_B has strong correlation.
        If big batch, correlation decrease so little problem.
        We could solve this problem through sampler (e.g. sampler + sampler reset).
        """
        # ignore label
        return (self.datasetA[n][0], self.datasetB[n][0])

    def __len__(self) -> int:
        return min(len(self.datasetA), len(self.datasetB))

if __name__ == "__main__":
    # test for clip
    i = torch.zeros(2, 2, 190, 200)
    o = pad_clip(i)
    print(o.size())