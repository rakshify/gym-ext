"""This implements a lookup table kind of the model."""

import os

from typing import Dict, Union

import numpy as np

from models.model import Model


class TableLookup(Model):
    """This implements a lookup table kind of the model."""
    name = 'table_lookup'
    
    def init_vars(self, nS: int, nA: int, **kwargs):
        self.q_vals = np.zeros((nS, nA))

    def predict(self, state: int):
        return self.q_vals[state, :]

    def update(self, update: Union[np.ndarray, float], **kwargs):
        si = kwargs.get("state")
        ai = kwargs.get("action")
        if si is None:
            if ai is None:
                self.q_vals = update
            else:
                self.q_vals[:, ai] = update
        else:
            if ai is None:
                self.q_vals[si, :] = update
            else:
                self.q_vals[si, ai] = update

    def save_dir(self, model_dir: str) -> Dict[str, str]:
        qvals_file = os.path.join(model_dir, "qvals.npy")
        with open(qvals_file, "wb") as f:
            np.save(f, qvals_file)
        return {"qvals": qvals_file}
