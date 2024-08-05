import os
import json

import torch

from .v1 import V1Model
from .parameters import V1Parameters


def load_model(root, model_id, decrease_gabba):

    def model_loader(hyperparams):
        model_params = hyperparams["model"]["params"]

        return V1Model(V1Parameters(**model_params), decrease_gabba)

    return _load_model(root, model_id, model_loader)


def _load_model(root, id, model_loader, device="cuda", dtype=torch.float):
    # model_loader: Create model instance from hyperparams
    model_path = os.path.join(root, id, "model.pt")
    model = model_loader(_load_hyperparams(root, id))
    model.load_state_dict(torch.load(model_path), strict=False)

    return model.to(device).type(dtype)


def _load_hyperparams(root, id):
    hyperparams_path = os.path.join(root, id, "hyperparams.json")
    with open(hyperparams_path) as f:
        hyperparams = json.load(f)

    return hyperparams