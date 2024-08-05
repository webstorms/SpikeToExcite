import os
import random

import torch
import torch.nn.functional as F
import pandas as pd

from src.dataset import PatchNaturalDataset
from src.models import load_model
from src.analysis import util
from src.analysis import example


class NeuroStimAnalysis:

    FIRING_THRESH = 0.7
    FLASH_HZ_RANGE = [1, 2, 4, 8, 16, 32, 64, 120]
    AMPLITUDES = [0, -0.01, -0.02, -0.04, -0.08, -0.16, -0.32, -0.64, -1.28]

    def __init__(self, root, model_id, metric, data_root="/home/datasets/natural", duration_ms=5000, decrease_gabba=0.8):
        self.stimulator = NeuroStim(root, model_id, data_root, duration_ms, decrease_gabba)

        if not os.path.exists(f"{root}/data/neurostimulation/{metric}.csv"):
            self._build(root, metric)

        self.metric_df = pd.read_csv(f"{root}/data/neurostimulation/{metric}.csv")

    def _build(self, root, metric, flash_ms=9, contrast=1, max_v=3.5, batch_size=512, seed=42, device="cuda"):
        data = []

        for flash_hz in NeuroStimAnalysis.FLASH_HZ_RANGE:
            for amp in NeuroStimAnalysis.AMPLITUDES:
                print(f"Building flash_hz={flash_hz} amp={amp}...")
                if metric == "firing_rate":
                    value = self.stimulator.compute_firing_rate_for_amp_and_freq(amp, flash_hz, flash_ms, contrast, max_v, batch_size, seed, device)[0].mean().item()
                elif metric == "frac_seizures":
                    value = self.stimulator.compute_fraction_of_seizures_for_amp_and_freq(amp, flash_hz, flash_ms, NeuroStimAnalysis.FIRING_THRESH, 0, contrast, max_v, batch_size, seed, device)[0]
                elif metric == "pred_mse":
                    value = self.stimulator.compute_mse_for_amp_and_freq(amp, flash_hz, flash_ms, contrast, max_v, batch_size, seed, device)[0]
                data.append({"flash_hz": flash_hz, "amp": amp, "value": value})

        metric_df = pd.DataFrame(data)
        metric_df.to_csv(f"{root}/data/neurostimulation/{metric}.csv", index=False)


class NeuroStim:

    def __init__(self, root, model_id, data_root="/home/datasets/natural", duration_ms=10000, decrease_gabba=0.8):
        self._val_dataset = PatchNaturalDataset(data_root, train=False, temp_len=int(duration_ms / 8.33), kernel=20, flip=False)
        self._healthy_model = load_model(f"{root}/data", model_id, 0.0)
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)
        self._duration_ms = duration_ms
        self.pokemon_clip = example.ExampleClipResponses.load_clip(root, "filtered_pokemon_tensor.pt")

    @staticmethod
    def gen_pulse_current(clip_ms, flash_ms, flash_hz, c=0.8):
        flash_clip = 0 * torch.ones(1, int(clip_ms/8.33), 1, 1)

        if flash_hz > 0:
            dt = int(120 / flash_hz)

            for i in range(0, flash_clip.shape[1], dt):
                for j in range(int(flash_ms/8.33)):
                    flash_clip[:, i+j] = c

        flash_clip = flash_clip.unsqueeze(0)
        flash_clip = flash_clip.repeat(1, 600, 1, 1, 1)

        return flash_clip

    def get_raster(self, model, flash_hz, amp, i, j, seed=42, device="cuda"):
        torch.manual_seed(seed)

        subspatialclip = self.pokemon_clip[:, :, i:i + 20, j:j + 20].unsqueeze(0).to(device)
        current = NeuroStim.gen_pulse_current(subspatialclip.shape[1] * 8.33, flash_ms=9, flash_hz=flash_hz, c=amp).to(device)

        loss = PredictiveLoss(warmup=5, crop=3, prediction_offset=0, reduction="mean")  # arguments used during model training
        seizure_model_output, healthy_model_output, seizure_model_spikes, healthy_model_spikes = self._get_model_outputs(subspatialclip, current, v=0)  # clips, current, v=0)

        if model == "control":
            mse = loss(healthy_model_output, subspatialclip)
            return healthy_model_spikes.cpu(), mse.cpu(), subspatialclip.cpu(), healthy_model_output.cpu()
        elif model == "seizure":
            mse = loss(seizure_model_output, subspatialclip)
            return seizure_model_spikes.cpu(), mse.cpu(), subspatialclip.cpu(), seizure_model_output.cpu()

    def compute_firing_rate_for_amp_and_freq(self, amp, freq, flash_ms=9, contrast=1, max_v=3.5, batch_size=512, seed=42, device="cuda"):

        def error_metric(clips, input_current):
            _, _, seizure_model_spikes, healthy_model_spikes = self._get_model_outputs(clips, input_current)

            return seizure_model_spikes, healthy_model_spikes

        all_scores = self._compute_metric_for_amp_and_freq(error_metric, amp, freq, flash_ms, contrast, max_v, batch_size, seed, device)
        seizure_model_firing_rate = torch.concat([batch_scores[0] for batch_scores in all_scores]).mean((0, 2))  # Mean firing per unit
        healthy_model_firing_rate = torch.concat([batch_scores[1] for batch_scores in all_scores]).mean((0, 2))  # Mean firing per unit

        return seizure_model_firing_rate.cpu(), healthy_model_firing_rate.cpu()

    def compute_fraction_of_seizures_for_amp_and_freq(self, amp, freq, flash_ms=9, firing_thresh=0.3, bins_thresh=5, contrast=1, max_v=3.5, batch_size=512, seed=42, device="cuda"):

        def error_metric(clips, input_current):
            _, _, seizure_model_spikes, healthy_model_spikes = self._get_model_outputs(clips, input_current)
            pop_seizure_model_spikes = seizure_model_spikes.mean(1).cpu()
            pop_healthy_model_spikes = healthy_model_spikes.mean(1).cpu()
            frac_seizures_in_seizure_model = ((pop_seizure_model_spikes > firing_thresh).sum(1) > bins_thresh).sum()
            frac_seizures_in_healthy_model = ((pop_healthy_model_spikes > firing_thresh).sum(1) > bins_thresh).sum()

            return frac_seizures_in_seizure_model, frac_seizures_in_healthy_model

        all_scores = self._compute_metric_for_amp_and_freq(error_metric, amp, freq, flash_ms, contrast, max_v, batch_size, seed, device)
        frac_seizures_in_seizure_model = torch.stack([batch_scores[0] for batch_scores in all_scores]).sum() / len(self._val_dataset)
        frac_seizures_in_healthy_model = torch.stack([batch_scores[1] for batch_scores in all_scores]).sum() / len(self._val_dataset)

        return frac_seizures_in_seizure_model.cpu().item(), frac_seizures_in_healthy_model.cpu().item()

    def compute_mse_for_amp_and_freq(self, amp, freq, flash_ms=9, contrast=1, max_v=3.5, batch_size=512, seed=42, device="cuda"):
        loss = PredictiveLoss(warmup=5, crop=3, prediction_offset=0, reduction="mean")  # arguments used during model training

        def error_metric(clips, input_current):
            seizure_model_output, healthy_model_output, _, _ = self._get_model_outputs(clips, input_current)
            seizure_model_mse = loss(seizure_model_output, clips)
            healthy_model_mse = loss(healthy_model_output, clips)

            return seizure_model_mse, healthy_model_mse

        all_scores = self._compute_metric_for_amp_and_freq(error_metric, amp, freq, flash_ms, contrast, max_v, batch_size, seed, device)
        seizure_model_mse = torch.stack([batch_scores[0] for batch_scores in all_scores]).mean()
        healthy_model_mse = torch.stack([batch_scores[1] for batch_scores in all_scores]).mean()

        return (100 * ((seizure_model_mse - healthy_model_mse) / healthy_model_mse)).cpu().item(), None

    def _compute_metric_for_amp_and_freq(self, error_metric, amp, freq, flash_ms=9, contrast=1, max_v=3.5, batch_size=512, seed=42, device="cuda"):
        random.seed(seed)
        torch.manual_seed(seed)

        input_current = NeuroStim.gen_pulse_current(self._duration_ms, flash_ms, freq, c=amp).to(device)

        def _error_metric(clips):
            clips = torch.clamp(contrast * clips, min=-max_v, max=max_v)

            return error_metric(clips, input_current)

        return util.compute_metric(self._val_dataset, _error_metric, batch_size=batch_size, device=device, dtype=torch.float)

    def _get_model_outputs(self, clips, current, v=0):
        clips = F.pad(clips, (0, 0, 0, 0, 14, 0), value=v)  # to ensure model output length = input length

        with torch.no_grad():
            healthy_model_output, healthy_model_spikes, _, _, _, _ = self._healthy_model(clips, "val")
            seizure_model_output, seizure_model_spikes, _, _, _, _ = self._seizure_model(clips, "val", current=current)

        return seizure_model_output, healthy_model_output, seizure_model_spikes, healthy_model_spikes


class PredictiveLoss:

    def __init__(self, warmup, crop, prediction_offset, reduction="mean"):
        self._warmup = warmup
        self._crop = crop
        self._prediction_offset = prediction_offset
        self._reduction = reduction

    def __call__(self, output, target):
        # Remove warmup frames and align output with target (due to encoder span creating misalignment)
        encoder_span_offset = target.shape[2] - output.shape[2]
        output = output[:, :, self._warmup:]
        target = target[:, :, self._warmup + encoder_span_offset:]
        assert output.shape[2] == target.shape[2]

        # Remove boarders
        if self._crop > 0:
            output = output[:, :, :, self._crop:-self._crop, self._crop:-self._crop]
            target = target[:, :, :, self._crop:-self._crop, self._crop:-self._crop]

        return F.mse_loss(output, target, reduction=self._reduction)
