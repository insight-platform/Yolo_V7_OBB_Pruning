from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np


class IDataset(ABC, Dataset):

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def classes_number(self) -> int:
        pass

    @property
    @abstractmethod
    def annotations(self) -> List[np.ndarray]:
        pass


class IDataloader(DataLoader):
    @property
    @abstractmethod
    def dataset(self) -> IDataset:
        pass

