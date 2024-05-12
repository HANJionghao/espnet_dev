from typing import Dict, Tuple, Union, Optional, Callable
import numpy as np

from espnet2.train.dataset import AbsDataset


class ESPnetEZDataset(AbsDataset):
    def __init__(
        self,
        dataset,
        data_info,
        preprocess: Optional[
            Callable[[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]
        ] = None,
    ):
        self.dataset = dataset
        self.data_info = data_info
        self.preprocess = preprocess

    def has_name(self, name) -> bool:
        return name in self.data_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.data_info.keys())

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict]:
        idx = int(uid)
        if self.preprocess is None:
            data = {k: v(self.dataset[idx]) for k, v in self.data_info.items()}
        else:
            # TODO(jhan): upon this change, existing espnetez demos should be updated with use_preprocess=False
            # when not using kaldi files as inputs; because when using existing datasets (not kaldi files),
            # the "preprocessor" part (e.g. tokenizations) is defined with data_info already

            data = self.preprocess(
                str(uid), {k: v(self.dataset[idx]) for k, v in self.data_info.items()}
            )
        return (str(uid), data)

    def __len__(self) -> int:
        return len(self.dataset)
