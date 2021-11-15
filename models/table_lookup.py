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

    def save_vars(self, model_dir: str) -> Dict[str, str]:
        qvals_file = os.path.join(model_dir, "qvals.npy")
        with open(qvals_file, "wb") as f:
            np.save(f, self.q_vals)
        return {"qvals": qvals_file}

    def load_vars(self, vars: Dict[str, str]):
        with open(vars["qvals"], "rb") as f:
            self.q_vals = np.load(f)
