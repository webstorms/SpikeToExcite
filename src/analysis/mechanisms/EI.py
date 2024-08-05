import random

import torch
import devtorch
import pandas as pd
from scipy.stats import mannwhitneyu
from brainbox.neural.correlation import cc
from brainbox.transforms import GaussianKernel

from src import dataset
from src.models import load_model


class EIAnalysis:

    def __init__(self, root, model_id, data_root="/home/datasets/natural", duration_ms=10000, decrease_gabba=0.8, seed=42, device="cuda"):
        healthy_model = load_model(f"{root}/data", model_id, 0.0)
        seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)

        provocative_clip_query = ProvocativeClipQuery(root, model_id, data_root, duration_ms, decrease_gabba, seed, device)
        if device == "cuda":
            torch.cuda.empty_cache()

        self.EI_stats_for_control_model_for_provocative_input = EIStatsForModelQuery(healthy_model, data_root, clip_idx=provocative_clip_query.seizure_clips_idx)
        self.EI_stats_for_seizure_model_for_provocative_input = EIStatsForModelQuery(seizure_model, data_root, clip_idx=provocative_clip_query.seizure_clips_idx)

        self.EI_stats_for_control_model_for_nonprovocative_input = EIStatsForModelQuery(healthy_model, data_root, clip_idx=provocative_clip_query.non_seizure_clips_idx)
        self.EI_stats_for_seizure_model_for_nonprovocative_input = EIStatsForModelQuery(seizure_model, data_root, clip_idx=provocative_clip_query.non_seizure_clips_idx)

    def get_EI_ratio_df(self):
        return self._data_to_df(
            self.EI_stats_for_control_model_for_provocative_input.get_neuron_EI(),
            self.EI_stats_for_seizure_model_for_provocative_input.get_neuron_EI(),
            self.EI_stats_for_control_model_for_nonprovocative_input.get_neuron_EI(),
            self.EI_stats_for_seizure_model_for_nonprovocative_input.get_neuron_EI(),
            "ratio")

    def get_EI_CC_df(self, n_bins=9):
        return self._data_to_df(
            self.EI_stats_for_control_model_for_provocative_input.get_smooth_cc(n_bins)[-1],
            self.EI_stats_for_seizure_model_for_provocative_input.get_smooth_cc(n_bins)[-1],
            self.EI_stats_for_control_model_for_nonprovocative_input.get_smooth_cc(n_bins)[-1],
            self.EI_stats_for_seizure_model_for_nonprovocative_input.get_smooth_cc(n_bins)[-1],
            "CC")

    def _data_to_df(self, control_model_provstim_EI_query, seizure_model_provstim_EI_query, control_model_stim_EI_query, seizure_model_stim_EI_query, metric_name):
        data = []

        for i in range(len(control_model_provstim_EI_query)):
            data.append({"model": "seizure", "stimulus": "non-provocative", metric_name: seizure_model_stim_EI_query[i].item()})
            data.append({"model": "healthy", "stimulus": "non-provocative", metric_name: control_model_stim_EI_query[i].item()})
            data.append({"model": "seizure", "stimulus": "provocative", metric_name: seizure_model_provstim_EI_query[i].item()})
            data.append({"model": "healthy", "stimulus": "provocative", metric_name: control_model_provstim_EI_query[i].item()})

        return pd.DataFrame(data)

    @staticmethod
    def plot_ratio_stats(EI_ratio_df):
        seizure_ratio_nonprovocative = EI_ratio_df[(EI_ratio_df["stimulus"] == "non-provocative") & (EI_ratio_df["model"] == "seizure")]["ratio"]
        control_ratio_nonprovocative = EI_ratio_df[(EI_ratio_df["stimulus"] == "non-provocative") & (EI_ratio_df["model"] == "healthy")]["ratio"]
        seizure_ratio_provocative = EI_ratio_df[(EI_ratio_df["stimulus"] == "provocative") & (EI_ratio_df["model"] == "seizure")]["ratio"]
        control_ratio_provocative = EI_ratio_df[(EI_ratio_df["stimulus"] == "provocative") & (EI_ratio_df["model"] == "healthy")]["ratio"]

        _, nonprovocative_p = mannwhitneyu(seizure_ratio_nonprovocative, control_ratio_nonprovocative, alternative="two-sided")
        _, provocative_p = mannwhitneyu(seizure_ratio_provocative, control_ratio_provocative, alternative="two-sided")

        print("EI ratio")
        print(f"Non-provocative: seizure median ({seizure_ratio_nonprovocative.median():.2f}) control median ({control_ratio_nonprovocative.median():.2f}) p={nonprovocative_p}")
        print(f"Provocative: seizure median ({seizure_ratio_provocative.median():.2f}) control median ({control_ratio_provocative.median():.2f}) p={provocative_p}")

    @staticmethod
    def plot_CC_stats(EI_CC_df):
        seizure_ratio_nonprovocative = EI_CC_df[(EI_CC_df["stimulus"] == "non-provocative") & (EI_CC_df["model"] == "seizure")]["CC"]
        control_ratio_nonprovocative = EI_CC_df[(EI_CC_df["stimulus"] == "non-provocative") & (EI_CC_df["model"] == "healthy")]["CC"]
        seizure_ratio_provocative = EI_CC_df[(EI_CC_df["stimulus"] == "provocative") & (EI_CC_df["model"] == "seizure")]["CC"]
        control_ratio_provocative = EI_CC_df[(EI_CC_df["stimulus"] == "provocative") & (EI_CC_df["model"] == "healthy")]["CC"]

        _, nonprovocative_p = mannwhitneyu(seizure_ratio_nonprovocative, control_ratio_nonprovocative, alternative="two-sided")
        _, provocative_p = mannwhitneyu(seizure_ratio_provocative, control_ratio_provocative, alternative="two-sided")

        print("EI CC")
        print(f"Non-provocative: seizure median ({seizure_ratio_nonprovocative.median():.2f}) control median ({control_ratio_nonprovocative.median():.2f}) p={nonprovocative_p}")
        print(f"Provocative: seizure median ({seizure_ratio_provocative.median():.2f}) control median ({control_ratio_provocative.median():.2f}) p={provocative_p}")


class ProvocativeClipQuery:
    """This class returns all the clip idxs with a seizure. Here we use the seizure model and simply check if the
    population firing rate mean is over a particular threshold."""

    FIRING_THRESH = 0.7
    BINS_THRESH = 20

    def __init__(self, root, model_id, data_root, duration_ms=10000, decrease_gabba=0.8, seed=42, device="cuda"):
        self._test_dataset = dataset.PatchNaturalDataset(data_root, train=False, temp_len=int(duration_ms / 8.33), kernel=20, flip=False)
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)

        out = self._build(seed, device)
        spikes = torch.cat([out[i] for i in range(len(out))])
        self.seizure_clips_idx = ((spikes.mean(1) > ProvocativeClipQuery.FIRING_THRESH).sum(1) > ProvocativeClipQuery.BINS_THRESH).cpu()
        self.non_seizure_clips_idx = ~self.seizure_clips_idx

    def _metric(self, output, target):
        return output

    def _build(self, seed, device):
        random.seed(seed)
        torch.manual_seed(seed)
        return devtorch.compute_metric(self._seizure_model, self._test_dataset, self._metric, batch_size=256, mode="just_spikes", device=device)


class EIStatsForModelQuery:
    """This class returns EI stats for a model."""

    def __init__(self, model, data_root, duration_ms=10000, clip_idx=None, seed=42, device="cuda"):
        self._test_dataset = dataset.PatchNaturalDataset(data_root, train=False, temp_len=int(duration_ms / 8.33), kernel=20, flip=False)
        self._model = model

        out = self._build(seed, device)
        self.ex_tensor = torch.cat([out[i][0] for i in range(len(out))])
        self.in_tensor = torch.cat([out[i][1] for i in range(len(out))])

        if clip_idx is not None:
            self.ex_tensor = self.ex_tensor[clip_idx]
            self.in_tensor = self.in_tensor[clip_idx]

    def get_smooth_cc(self, n_bins=9):
        gk = GaussianKernel(n_bins, 101)  # 9 bins is approx 72ms
        smooth_ex = gk(self.ex_tensor.permute(1, 0, 2).flatten(1, 2).unsqueeze(0))[0]
        smooth_in = gk(self.in_tensor.permute(1, 0, 2).flatten(1, 2).unsqueeze(0))[0]
        smooth_cc = cc(smooth_ex, -smooth_in)

        return smooth_ex, smooth_in, smooth_cc

    def get_raw_cc(self):
        raw_ex = self.ex_tensor.permute(1, 0, 2).flatten(1, 2).unsqueeze(0)[0]
        raw_in = self.in_tensor.permute(1, 0, 2).flatten(1, 2).unsqueeze(0)[0]
        raw_cc = cc(raw_ex, raw_in)

        return raw_ex, raw_in, raw_cc

    def get_neuron_EI(self):
        neuron_ex_sum = self.ex_tensor.permute(1, 0, 2).flatten(1, 2).sum(1)
        neuron_in_sum = self.in_tensor.permute(1, 0, 2).flatten(1, 2).abs().sum(1)

        # returns tensor of shape [neurons] (i.e. the EI ratio of ever neuron)
        return neuron_ex_sum / neuron_in_sum

    def _metric(self, output, target):
        output, spikes, mem, ex_rec, in_rec, input_current = output

        fwd_current = input_current[..., 0, 0].detach().cpu()
        fwd_ex_current = fwd_current.clone()
        fwd_ex_current[fwd_ex_current < 0] = 0
        ex_current = ex_rec.detach().cpu() + fwd_ex_current

        fwd_in_current = fwd_current.clone()
        fwd_in_current[fwd_in_current > 0] = 0
        in_current = in_rec.detach().cpu() + fwd_in_current

        return ex_current, in_current

    def _build(self, seed, device):
        random.seed(seed)
        torch.manual_seed(seed)
        out = devtorch.compute_metric(self._model, self._test_dataset, self._metric, batch_size=256, mode="val", device=device)

        return out
