from functools import partial
from types import MethodType

import lightning
import torch


def custom_repr(self, original_repr):
    shape_str = ""
    if hasattr(self, "shape"):
        shape_str = f":{tuple(self.shape)}"
    elif hasattr(self, "__len__"):
        shape_str = f":{len(self)}"
    return f"{{{self.__class__.__name__}{shape_str}}} {original_repr(self)}"


class CustomTensorReprCallback(lightning.Callback):
    def setup(self, *args, **kwargs) -> None:
        torch.Tensor.__repr__ = MethodType(
            partial(custom_repr, original_repr=torch.Tensor.__repr__), torch.Tensor
        )


if __name__ == '__main__':
    torch.Tensor.__repr__ = MethodType(
        partial(custom_repr, original_repr=torch.Tensor.__repr__), torch.Tensor
    )
    import numpy as np
    a = torch.rand([3,4,5])
    b = np.array([[1,2,3],[4,5,6]])
