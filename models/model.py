"""This implements base model class."""

from typing import Any, Dict, Union

import numpy as np


class Model(object):
    """This implements base model class."""
    name = ''
    
    def init_vars(self, num_features: int=None, nA: int=None, **kwargs):
        raise NotImplementedError("No variables to initialize for base model.")

    def predict(self, state: Any) -> Any:
        raise NotImplementedError("Base model can not predict.")

    def update(self, update: Union[np.ndarray, float], **kwargs):
        raise NotImplementedError("Base model can not update.")

    def serialize(self) -> Dict[str, str]:
        return {"name": self.name}
