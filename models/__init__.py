from models.model import Model
from models.linear import Linear
from models.table_lookup import TableLookup

_ALL_MODELS = [Linear, TableLookup]
_REGISTERED_MODELS = {m.name: m for m in _ALL_MODELS}


def get_model_by_name(name: str):
    if name not in _REGISTERED_MODELS:
        raise ValueError(f"Model {name} not found")
    return _REGISTERED_MODELS[name]
