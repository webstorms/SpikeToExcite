import random

import torch
import torch.nn.functional as F
import numpy as np
from devtorch import util


def get_model_outputs(clip, control_model, seizure_model, seed=42, v=0):
    random.seed(seed)
    torch.manual_seed(seed)

    clip = F.pad(clip, (0, 0, 0, 0, 14, 0), value=v)  # to ensure model output length = input length

    with torch.no_grad():
        c_output, c_spikes, c_mem, c_ex_rec_current, c_in_rec_current, c_input_current = control_model(clip.unsqueeze(0).cuda(), "val")
        s_output, s_spikes, s_mem, s_ex_rec_current, s_in_rec_current, s_input_current = seizure_model(clip.unsqueeze(0).cuda(), "val")

    return c_spikes[0].cpu(), s_spikes[0].cpu()


def spike_tensor_to_points(spikes):
    x = np.array([p[1].item() for p in torch.nonzero(spikes.cpu())])
    y = np.array([p[0].item() for p in torch.nonzero(spikes.cpu())])

    return x, y


def compute_metric(dataset, metric, batch_size=128, device="cuda", dtype=torch.float):
    metric_list = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)

    with torch.no_grad():
        for data, target in data_loader:
            data = util.cast(data, device, dtype)

            metric_value = metric(data)
            metric_list.append(metric_value)

    return metric_list
