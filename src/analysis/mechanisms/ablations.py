import random

import torch
import devtorch
import pandas as pd
from brainbox.neural.correlation import cc
from brainbox.transforms import GaussianKernel

from src import dataset
from src.models import load_model
from src.analysis import util
from src.analysis.mechanisms import EI


class ClipAblationAnalysis:
    """Class that builds results to check seizure occurrence in seizure model when provocative clips are blanked
    after seizure onset."""

    FIRING_THRESH = 0.7

    def __init__(self, root, model_id, data_root, duration_ms=10000, decrease_gabba=0.8, seed=42, device="cuda"):
        # Here we get all clips to which the seizure model has seizures
        provocative_clip_query = EI.ProvocativeClipQuery(root, model_id, data_root, duration_ms=duration_ms, decrease_gabba=decrease_gabba, seed=seed, device=device)
        seizure_clips = SeizureClipsTensorBuilder(data_root, clip_idx=provocative_clip_query.seizure_clips_idx).clips

        # Seizure model responses to all provocative clips
        self.full_clip_responses = SeizureModelResponseBuilder(root, model_id, SeizureClipDataset(seizure_clips), decrease_gabba=decrease_gabba, seed=seed, device=device)

        # Here we blank the provocative clips following seizure onset and obtain the seizure model responses to these clips
        self.ablated_clips = self._get_ablated_clips(seizure_clips, self.full_clip_responses)
        self.ablated_clip_responses = SeizureModelResponseBuilder(root, model_id, SeizureClipDataset(self.ablated_clips), decrease_gabba=decrease_gabba, seed=seed, device=device)

    def _get_ablated_clips(self, clips, full_clip_responses):
        seizure_bin_start = (full_clip_responses.spikes.mean(1) > ClipAblationAnalysis.FIRING_THRESH).int().argmax(dim=1)
        blanked_clips = clips.clone()

        for i in range(blanked_clips.shape[0]):
            blanked_clips[i, 0, seizure_bin_start[i]+4:] *= 0

        return blanked_clips

    def get_seizure_clip_fraction(self):
        n_clips = dataset.PatchNaturalDataset.TEST_LENGTH
        n_clips_with_seizures = self.ablated_clips.shape[0]
        n_ablated_clips_with_seizures = ((self.ablated_clip_responses.spikes.mean(1).cpu() > ClipAblationAnalysis.FIRING_THRESH).sum(1) > 0).sum().item()

        frac_of_clips_with_seizures = n_clips_with_seizures / n_clips
        frac_of_clips_with_seizures_following_ablation = n_ablated_clips_with_seizures / n_clips

        return frac_of_clips_with_seizures, frac_of_clips_with_seizures_following_ablation


class SeizureClipsTensorBuilder:
    """This class returns seizure clips as tensor from the PatchNaturalDataset."""

    def __init__(self, data_root, duration_ms=10000, clip_idx=None, seed=42, device="cuda"):
        assert clip_idx is not None
        self._test_dataset = dataset.PatchNaturalDataset(data_root, train=False, temp_len=int(duration_ms / 8.33), kernel=20, flip=False)

        out = self._build(seed, device)
        clips = torch.cat([out[i] for i in range(len(out))])
        self.clips = clips[clip_idx]

    def _build(self, seed, device):
        random.seed(seed)
        torch.manual_seed(seed)

        return util.compute_metric(self._test_dataset, lambda x: x, batch_size=256, device=device)


class SeizureClipDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for the seizure clips."""

    def __init__(self, clips):
        self._clips = clips

    def __getitem__(self, i):
        # Return tuple as this is what the model expects
        return self._clips[i], self._clips[i]

    def __len__(self):
        return self._clips.shape[0]


class SeizureModelResponseBuilder:
    """Responses of seizure model to clips."""

    def __init__(self, root, model_id, dataset, decrease_gabba=0.8, seed=42, device="cuda"):
        self._dataset = dataset
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)

        out = self._build(seed, device)
        self.spikes = torch.cat([out[i] for i in range(len(out))])

    def _metric(self, output, target):
        return output

    def _build(self, seed, device):
        random.seed(seed)
        torch.manual_seed(seed)
        return devtorch.compute_metric(self._seizure_model, self._dataset, self._metric, batch_size=256, mode="just_spikes", device=device)