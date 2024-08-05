import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import torch
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from brainbox import spiking

from src.models import load_model
from src.analysis import neurostim, tuning
from src.analysis.mechanisms import hyperbolic


class FiringRateVsContrastQuery:

    def __init__(self, root, model_id, data_root, contrast_values, duration_ms=5000, decrease_gabba=0.8):
        self._contrast_values = contrast_values
        self._dbs = neurostim.NeuroStim(root, model_id, data_root=data_root, duration_ms=duration_ms, decrease_gabba=decrease_gabba)

        if not os.path.exists(f"{root}/data/mechanisms/fr_vs_contrast.csv"):
            self._build(root)

        self.fr_vs_contrast_df = pd.read_csv(f"{root}/data/mechanisms/fr_vs_contrast.csv")

    def _build(self, root):
        data = []

        for contrast in self._contrast_values:
            seizure_model_firing_rate, healthy_model_firing_rate = self._dbs.compute_firing_rate_for_amp_and_freq(amp=0.0, freq=100, flash_ms=9, contrast=contrast, max_v=3.5, seed=42)

            for i in range(len(seizure_model_firing_rate)):
                data.append({"contrast": contrast, "model": "seizure", "unit": i, "fr": seizure_model_firing_rate[i].item()})
                data.append({"contrast": contrast, "model": "healthy", "unit": i, "fr": healthy_model_firing_rate[i].item()})

        fr_vs_contrast_df = pd.DataFrame(data)
        fr_vs_contrast_df.to_csv(f"{root}/data/mechanisms/fr_vs_contrast.csv", index=False)


class SeizureRateVsContrastQuery:

    FIRING_THRESH = 0.7
    BINS_THRESH = 0

    def __init__(self, root, model_id, data_root, contrast_values, duration_ms=5000, decrease_gabba=0.8):
        self._contrast_values = contrast_values
        self._dbs = neurostim.NeuroStim(root, model_id, data_root=data_root, duration_ms=duration_ms, decrease_gabba=decrease_gabba)

        if not os.path.exists(f"{root}/data/mechanisms/sr_vs_contrast.csv"):
            self._build(root)

        self.sr_vs_contrast_df = pd.read_csv(f"{root}/data/mechanisms/sr_vs_contrast.csv")

    def _build(self, root):
        data = []

        for contrast in self._contrast_values:
            frac_seizures_in_seizure_model, frac_seizures_in_healthy_model = self._dbs.compute_fraction_of_seizures_for_amp_and_freq(amp=0.0, freq=100, flash_ms=9, firing_thresh=SeizureRateVsContrastQuery.FIRING_THRESH, bins_thresh=SeizureRateVsContrastQuery.BINS_THRESH, contrast=contrast, max_v=3.5, seed=42)

            data.append({"contrast": contrast, "model": "seizure", "sr": frac_seizures_in_seizure_model})
            data.append({"contrast": contrast, "model": "healthy", "sr": frac_seizures_in_healthy_model})

        sr_vs_contrast_df = pd.DataFrame(data)
        sr_vs_contrast_df.to_csv(f"{root}/data/mechanisms/sr_vs_contrast.csv", index=False)


class ContrastResponseAnalyses:

    def __init__(self, root, model_id, contrast_values, decrease_gabba=0.8, seed=42):
        self._contrast_values = contrast_values
        self._healthy_model = load_model(f"{root}/data", model_id, 0.0)
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)
        # We use the preferred gratings from the healthy model, just like in Atallah et al. (2012)
        self._grating_query = tuning.GratingQuery(root, "healthy_model", self._healthy_model)

        if not os.path.exists(f"{root}/data/mechanisms/contrast_responses/m.pt"):
            self._build(root, contrast_values, seed)

        self.m = torch.load(f"{root}/data/mechanisms/contrast_responses/m.pt")
        self.c = torch.load(f"{root}/data/mechanisms/contrast_responses/c.pt")
        self.healthy_response = torch.load(f"{root}/data/mechanisms/contrast_responses/healthy_p.pt")
        self.seizure_response = torch.load(f"{root}/data/mechanisms/contrast_responses/seizure_p.pt")

    def get_contrast_response_data(self, i):
        # Mean and SEM of responses to increasing contrast
        healthy_sem_df = self._contrast_responses_to_df(self.healthy_response[i])
        seizure_sem_df = self._contrast_responses_to_df(self.seizure_response[i])

        # Contrast values
        min_v = self._contrast_values[0]
        max_v = self._contrast_values[-1]
        c = 100 * (torch.Tensor(self._contrast_values) - min_v) / (max_v - min_v)

        # Hyperbolic fits
        healthy_response = self.healthy_response[i].mean(0)
        seizure_response = self.seizure_response[i].mean(0)

        _, healthy_response_fit = hyperbolic.get_fitted_hyperbolic_function(c, healthy_response)
        c, seizure_response_fit = hyperbolic.get_fitted_hyperbolic_function(c, seizure_response)
        c = ((c / 100) * (self._contrast_values[-1] - self._contrast_values[0])) + self._contrast_values[0]

        return c, healthy_response_fit, seizure_response_fit, healthy_sem_df, seizure_sem_df

    def get_change_in_contrast_response_data(self):
        query_idxs = self._get_change_in_contrast_response_data_valid_idxs()
        healthy_v = self.healthy_response.mean(1)[query_idxs]
        seizure_v = self.seizure_response.mean(1)[query_idxs]

        return healthy_v, seizure_v

    def get_linear_fit_params(self):
        query_idxs = self._get_change_in_contrast_response_data_valid_idxs()

        return self.m[query_idxs], self.c[query_idxs]

    def print_linear_fit_params(self):
        m, c = self.get_linear_fit_params()
        m = m.mean().item()
        c = c.mean().item()
        print(f"m={m} c={c}")

    def print_m_stats(self):
        query_idxs = self._get_change_in_contrast_response_data_valid_idxs()
        t_stat, p_value = stats.ttest_1samp(self.m[query_idxs], 1)

        print(f"t_stat={t_stat} p_value={p_value}")

    def _get_change_in_contrast_response_data_valid_idxs(self):
        tuning_df = self._grating_query.filtered_tuning_results
        tuning_query = (tuning_df.index >= 90) & (tuning_df["F1F0"] > 1)

        return tuning_df[tuning_query].index

    def _build(self, root, contrast_values, seed):
        m_list = []
        c_list = []
        healthy_p_list = []
        seizure_p_list = []

        # probe
        torch.manual_seed(seed)
        n_units = self._healthy_model.hyperparams["params"]["n_in"]

        for i in range(n_units):
            m, c, healthy_response, seizure_response = self._probe_for_unit(i, contrast_values)
            m_list.append(m)
            c_list.append(c)
            healthy_p_list.append(healthy_response)
            seizure_p_list.append(seizure_response)

        # Save tensors
        torch.save(torch.cat(m_list), f"{root}/data/mechanisms/contrast_responses/m.pt")
        torch.save(torch.cat(c_list), f"{root}/data/mechanisms/contrast_responses/c.pt")
        torch.save(torch.stack(healthy_p_list), f"{root}/data/mechanisms/contrast_responses/healthy_p.pt")
        torch.save(torch.stack(seizure_p_list), f"{root}/data/mechanisms/contrast_responses/seizure_p.pt")

    def _probe_for_unit(self, i, contrast_values):
        clips = self._get_contrast_gratings_for_unit(i, contrast_values=contrast_values, max_v=1)
        healthy_response = ContrastResponseAnalyses.input_to_spikes(self._healthy_model, clips).cpu()[0, :, :, i].mean(-1)
        seizure_response = ContrastResponseAnalyses.input_to_spikes(self._seizure_model, clips).cpu()[0, :, :, i].mean(-1)

        # Linear fit to obtain m and c between healthy response to seizure response for increasing in contrast
        reg = LinearRegression().fit(healthy_response.mean(0).unsqueeze(1).numpy(), seizure_response.mean(0).unsqueeze(1).numpy())

        return torch.Tensor([reg.coef_[0]]), torch.Tensor([reg.intercept_[0]]), healthy_response, seizure_response

    def _get_contrast_gratings_for_unit(self, i, contrast_values, max_v):
        clip_list = []
        for c in contrast_values:
            grating_clip = self._grating_query.query._gratings[i].clone()
            new_clip = torch.clamp(c * grating_clip, -max_v, max_v)

            clip_list.append(new_clip)

        return torch.stack(clip_list)

    def _contrast_responses_to_df(self, contrast_responses):
        data = []
        for i in range(contrast_responses.shape[1]):
            for j in range(contrast_responses.shape[0]):
                data.append({"c": self._contrast_values[i], "trial": j, "response": contrast_responses[j][i].item()})

        return pd.DataFrame(data)

    @staticmethod
    def input_to_spikes(model, data, n_trials=5):
        with torch.no_grad():
            rate_trains_list = []
            data = data.unsqueeze(1).cuda()

            data *= 1

            b, _, _, _, _ = data.shape
            data = data.repeat(n_trials, 1, 1, 1, 1)
            spike_trains = model(data, mode="just_spikes")

            _, n, t = spike_trains.shape
            spike_trains = spike_trains.view(n_trials, b, n, t)
            rate_trains_list.append(spike_trains[:, :, :, 26:])

            return torch.stack(rate_trains_list)
