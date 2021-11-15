"""This implements base model class."""

from typing import Any, Union

import numpy as np


class Model(object):
    """This implements base model class."""
    name = ''
    
    def init_vars(self, nS: int, nA: int, **kwargs):
        raise NotImplementedError("No variables to initialize for base model.")

    def predict(self, state: Any):
        raise NotImplementedError("Base model can not predict.")

    def update(self, update: Union[np.ndarray, float], **kwargs):
        raise NotImplementedError("Base model can not update.")

    def serialize(self):
        return {"name": self.name}
