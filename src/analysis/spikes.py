import random

import torch
import numpy as np
import pandas as pd
from brainbox import spiking

from src import dataset
from src.models import load_model


class SpikeAnalysis:

    def __init__(self, root, model_id, decrease_gabba=0.8, firing_tresh=0.7, dataset_name="starwars"):
        self._control_model = load_model(f"{root}/data", model_id, 0.0)
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)

        self.control_spike_tensor_builder = SpikeTensorBuilder(root, self._control_model, dataset_name)
        self.seizure_spike_tensor_builder = SpikeTensorBuilder(root, self._seizure_model, dataset_name)

        self.seizure_clip_idx = ((self.seizure_spike_tensor_builder.spike_tensor.mean(1) > firing_tresh).sum(1) > 0)

        self.control_model_provocative_clips_spike_stats = SpikeStats(self.control_spike_tensor_builder.spike_tensor[self.seizure_clip_idx])
        self.control_model_nonprovocative_clips_spike_stats = SpikeStats(self.control_spike_tensor_builder.spike_tensor[~self.seizure_clip_idx])
        self.seizure_model_provocative_clips_spike_stats = SpikeStats(self.seizure_spike_tensor_builder.spike_tensor[self.seizure_clip_idx])
        self.seizure_model_nonprovocative_clips_spike_stats = SpikeStats(self.seizure_spike_tensor_builder.spike_tensor[~self.seizure_clip_idx])


class SpikeTensorBuilder:

    def __init__(self, root, model, dataset_name="starwars", data_root="/home/datasets/natural"):
        if dataset_name == "starwars":
            self.test_dataset = dataset.PatchStarwarsDataset(root=root)
        elif dataset_name == "natural":
            # Here we use 4 sec rather than 10 sec clip due to hitting memory hits with the full clip length
            self.test_dataset = dataset.PatchNaturalDataset(data_root, train=False, temp_len=int(4000 / 8.33), kernel=20, flip=False)

        self.model = model
        self.clips = []
        self.spike_tensor = self.build_spike_tensor()
        self.clips = torch.stack(self.clips)

    def build_spike_tensor(self, batch_size=1000, seed=142):
        random.seed(seed)
        torch.manual_seed(seed)
        spikes_list = []
        x_list = []

        for i in range(len(self.test_dataset)):
            x, y = self.test_dataset[i]
            self.clips.append(x)
            x_list.append(x)

            if len(x_list) == batch_size:
                x_batch = torch.cat(x_list)
                x_list = []

                with torch.no_grad():
                    spikes = self.model(x_batch.unsqueeze(1).cuda(), mode="just_spikes")
                    spikes = spikes.cpu().detach()
                    spikes_list.append(spikes)

        spike_tensor = torch.cat(spikes_list)

        return spike_tensor

    def get_raster_coo(self, clip_idx):
        def spike_tensor_to_points(spike_tensor):
            x = np.array([p[1].item() for p in torch.nonzero(spike_tensor.cpu())])
            y = np.array([p[0].item() for p in torch.nonzero(spike_tensor.cpu())])

            return x, y

        return spike_tensor_to_points(self.spike_tensor[clip_idx])


class SpikeStats:

    def __init__(self, spike_tensor, dt=1000/120):
        self.spike_tensor = spike_tensor
        self.duration_ms = (self.spike_tensor.shape[2] * dt)

    def get_firing_rate(self, max_rate=100, loglog=False, binned=True):
        spike_counts = self.spike_tensor[:, :].sum(2)
        firing_rates = (spike_counts / self.duration_ms) * 1000

        if not binned:
            in_firing = firing_rates[:, :90].flatten()
            ex_firing = firing_rates[:, 90:].flatten()

            data_list = []

            for v in in_firing:
                data_list.append({"type": "In", "v": v.item()})

            for v in ex_firing:
                data_list.append({"type": "Ex", "v": v.item()})

            return pd.DataFrame(data_list)

        firing_rates[firing_rates > max_rate] = max_rate
        firing_rates_x = np.histogram(firing_rates.flatten(), list(range(0, max_rate)))[1]
        firing_rates_y = np.histogram(firing_rates.flatten(), list(range(0, max_rate+1)))[0]

        firing_rates_y = firing_rates_y / firing_rates_y.sum()

        if loglog:
            firing_rates_y = np.log10(firing_rates_y)

        return firing_rates_x, firing_rates_y

    def get_synchronization(self, pairs=400, seed=0):
        torch.manual_seed(seed)
        cross_covariance_tensor = spiking.compute_synchronization(self.spike_tensor, pairs, bin_dt=25)
        synchronization_df = spiking.compute_synchronization_df(cross_covariance_tensor.mean(0))

        mean_synchronization = synchronization_df.groupby("lag").mean()
        lag = mean_synchronization.index / 1000
        correlation = mean_synchronization["correlation"].values

        return lag.values, correlation


def get_monkey_firing_rates(root):
    firing_rates_df = pd.read_csv(f"{root}/data/spike/exp_firing.csv")
    firing_rates = firing_rates_df.iloc[:, 0]
    probability = firing_rates_df.iloc[:, 1]
    probability = probability / probability.sum()  # Normalize
    probability = np.log10(probability)

    return firing_rates.values, probability.values


def get_monkey_synchronization(root):
    cv_isi_df = pd.read_csv(f"{root}/data/spike/exp_synchronization.csv")
    lag = cv_isi_df.iloc[:, 0]
    correlation = cv_isi_df.iloc[:, 1]

    return lag.values, correlation.values
