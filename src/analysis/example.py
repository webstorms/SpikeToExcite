import random

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

from src import dataset, models
from src.analysis import util


class ExampleClipResponses:

    def __init__(self, root, model_id, data_root="/home/datasets/natural", duration_ms=10000, decrease_gabba=0.8):
        self.duration_ms = duration_ms
        self.control_model = models.load_model(f"{root}/data", model_id, 0.0)
        self.seizure_model = models.load_model(f"{root}/data", model_id, decrease_gabba)

        # Load known provocative stimuli
        self.pokemon_clip = ExampleClipResponses.load_clip(root, "filtered_pokemon_tensor.pt")
        self.all_the_lights_clip = ExampleClipResponses.load_clip(root, "filtered_all_the_lights_tensor.pt")
        self.citroen_clip = ExampleClipResponses.load_clip(root, "filtered_citroen_tensor.pt")
        self.incredibles_clip = ExampleClipResponses.load_clip(root, "filtered_incredibles_tensor.pt")
        self.take_my_breath_clip = ExampleClipResponses.load_clip(root, "filtered_take_my_breath_tensor.pt")

        self.test_dataset = dataset.PatchNaturalDataset(data_root, train=False, temp_len=int(duration_ms / 8.33), kernel=20, flip=False)
        self.nonprov_clip = self.test_dataset._dataset[..., ::2, ::2][1]

    @staticmethod
    def load_clip(root, name):
        clip = torch.load(f"{root}/data/vids/{name}").float()
        clip = (clip - clip.mean()) / clip.std()
        clip = torch.clamp(clip, min=-3.5, max=3.5)
        clip = clip.unsqueeze(1)
        clip = clip.repeat_interleave(4, dim=0)  # To match hz the model was trained at
        clip = clip[:1200, 0, ::2, ::2].unsqueeze(0)

        return clip

    @staticmethod
    def get_specgram_data(response, NFFT=12, Fs=120, noverlap=4):
        power, freqs, bins, _ = matplotlib.pyplot.specgram(response, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
        power = np.log10(power)

        return power, freqs, bins

    def get_pokemon_clip_responses(self, i, j, seed=42):
        subspatialclip = self.pokemon_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)

    def get_all_the_lights_clip_responses(self, i, j, seed=42):
        subspatialclip = self.all_the_lights_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)

    def get_citroen_clip_responses(self, i, j, seed=42):
        subspatialclip = self.citroen_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)

    def get_incredibles_clip_responses(self, i, j, seed=42):
        subspatialclip = self.incredibles_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)

    def get_take_my_breath_clip_responses(self, i, j, seed=42):
        subspatialclip = self.take_my_breath_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)

    def get_nonprov_movie_responses(self, i, j, seed=42):
        subspatialclip = self.nonprov_clip[:, :, i:i + 20, j:j + 20]
        return subspatialclip, *util.get_model_outputs(subspatialclip, self.control_model, self.seizure_model, seed, v=0)


class InhibDecreaseFRCalculator:

    def __init__(self, root, model_id, data_root="/home/datasets/natural"):
        self.pokemon_fr_list = []
        self.all_the_lights_fr_list = []
        self.citroen_fr_list = []
        self.incredibles_fr_list = []
        self.take_my_breath_fr_list = []

        for decrease_gabba in np.linspace(0, 1, 11):
            seizure_example = ExampleClipResponses(root, model_id, data_root=data_root, duration_ms=5000, decrease_gabba=decrease_gabba)

            # Pokemon
            _, control_model_response, seizure_model_response = seizure_example.get_pokemon_clip_responses(i=-31, j=20, seed=42)
            self.pokemon_fr_list.append(120 * seizure_model_response.mean(0).max().item())

            # All the lights
            _, control_model_response, seizure_model_response = seizure_example.get_all_the_lights_clip_responses(i=-31, j=20, seed=42)
            self.all_the_lights_fr_list.append(120 * seizure_model_response.mean(0).max().item())

            # Citroen
            _, control_model_response, seizure_model_response = seizure_example.get_citroen_clip_responses(i=-31, j=20, seed=42)
            self.citroen_fr_list.append(120 * seizure_model_response.mean(0).max().item())

            # Citroen
            _, control_model_response, seizure_model_response = seizure_example.get_incredibles_clip_responses(i=-31, j=20, seed=42)
            self.incredibles_fr_list.append(120 * seizure_model_response.mean(0).max().item())

            # Take my breath
            _, control_model_response, seizure_model_response = seizure_example.get_take_my_breath_clip_responses(i=20, j=20, seed=42)
            self.take_my_breath_fr_list.append(120 * seizure_model_response.mean(0).max().item())