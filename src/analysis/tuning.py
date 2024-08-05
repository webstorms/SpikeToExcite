import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from brainbox import spiking
from brainbox import tuning

from src.models import load_model


class GratingAnalysis:

    def __init__(self, root, model_id, decrease_gabba=0.8):
        control_model = load_model(f"{root}/data", model_id, 0.0)
        seizure_model = load_model(f"{root}/data", model_id, decrease_gabba=decrease_gabba)

        self.control_grating_query = GratingQuery(root, "control_model", control_model)
        self.seizure_grating_query = GratingQuery(root, "seizure_model", seizure_model)

    def get_mean_responses_to_gratings(self):
        df = self.tuning_df
        control_model_response = df[df["model"] == "Control model"]["mean_response"].values
        seizure_model_response = df[df["model"] == "Seizure model"]["mean_response"].values

        return control_model_response, seizure_model_response

    def print_stats_table(self):
        tuning_stat_list = []

        for measure in ["OSI", "DSI"]:
            for ex in [True, False]:
                control_query = (self.tuning_df["model"] == "Control model") & (self.tuning_df["ex"] == ex)
                seizure_query = (self.tuning_df["model"] == "Seizure model") & (self.tuning_df["ex"] == ex)
                control_vs = self.tuning_df[control_query][measure].values
                seizure_vs = self.tuning_df[seizure_query][measure].values
                U1, p = mannwhitneyu(control_vs, seizure_vs, alternative="two-sided")
                unit_type = "Excitatory" if ex else "Inhibitory"
                control_str = f"${np.mean(control_vs):.2f} \pm {np.std(control_vs):.2f}$"
                seizure_str = f"${np.mean(seizure_vs):.2f} \pm {np.std(seizure_vs):.2f}$"
                tuning_stat_list.append({"metric": measure, "Unit type": unit_type, "Control model": control_str, "Seizure model": seizure_str, "p-value": f"${p:.4f}$"})

        print(pd.DataFrame(tuning_stat_list).set_index(["metric", "Unit type"]).to_latex())

    @property
    def tuning_df(self):
        control_model_tuning_df = self.control_grating_query.filtered_tuning_results
        seizure_model_tuning_df = self.seizure_grating_query.filtered_tuning_results
        intersection_index = control_model_tuning_df.index.intersection(seizure_model_tuning_df.index)
        control_model_tuning_df = control_model_tuning_df.loc[intersection_index]
        seizure_model_tuning_df = seizure_model_tuning_df.loc[intersection_index]
        control_model_tuning_df["model"] = "Control model"
        seizure_model_tuning_df["model"] = "Seizure model"

        return pd.concat([control_model_tuning_df, seizure_model_tuning_df])


class GratingQuery:

    def __init__(self, root, model_name, model):
        self._root = root
        self._model_name = model_name
        self._model = model

        if not os.path.exists(f"{self.tuning_path}/probe.csv"):
            self.probe()

        self.query = tuning.TuningQuery(self.tuning_path)
        self.filtered_tuning_results = self.query.validate(response_threshold=0.1, fit_threshold=0, additional_data={"ex": self._model._neurons.excitatory_idx.cpu().detach().numpy()})

    @property
    def tuning_path(self):
        Path(f"{self._root}/data/tuning/{self._model_name}").mkdir(parents=True, exist_ok=True)
        return f"{self._root}/data/tuning/{self._model_name}"

    @staticmethod
    def input_to_spikes(model, data, dt=1000/120, n_trials=4):
        with torch.no_grad():
            rate_trains_list = []
            data = data.unsqueeze(1).cuda()

            data *= 1

            b, _, _, _, _ = data.shape
            data = data.repeat(n_trials, 1, 1, 1, 1)
            spike_trains = model(data, mode="just_spikes")

            _, n, t = spike_trains.shape
            spike_trains = spike_trains.view(n_trials, b, n, t)
            spike_trains = spike_trains.mean(0)

            rate_trains = spiking.rate.bin_spikes(spike_trains, dt, 400, pad_input=True, gaussian=True, sigma=9)  # 9 bins is approx 72ms
            rate_trains_list.append(rate_trains[:, :, 26:])

            mean_rate_trains = torch.stack(rate_trains_list).mean(dim=0)
            return mean_rate_trains

    def probe(self):
        torch.manual_seed(42)

        probe_ms = 3000  # Probe for 3s
        dt = 1000 / 120
        warmup_period = 15 + 26  # Same as used for training

        thetas = np.linspace(0, np.pi*2, 72)
        spatial_freqs = np.around(np.linspace(0.01, 0.2, 10), 4)
        temporal_freqs = [1, 2, 4, 8]

        input_to_spikes = lambda data: GratingQuery.input_to_spikes(self._model, data, dt)
        gratings = tuning.GratingsProber(input_to_spikes, amplitude=1, rf_w=20, rf_h=20, duration=probe_ms+warmup_period*dt, dt=dt, thetas=thetas, spatial_freqs=spatial_freqs, temporal_freqs=temporal_freqs)
        gratings.probe_and_fit(self.tuning_path, probe_batch=128, response_batch=32)

