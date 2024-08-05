import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import devtorch

from .snn import SNN


class V1Model(devtorch.DevModel):

    def __init__(self, params, decrease_gabba=0.0):
            super().__init__()
            self.params = params

            self._encoder_bias = nn.Parameter(torch.rand(params.n_in))
            self._decoder_bias = nn.Parameter(torch.rand(1))
            self._encoder_weight = nn.Parameter(torch.rand(params.n_in, 1, params.encoder_span, params.rf_size, params.rf_size), requires_grad=True)
            self._decoder_weight = nn.Parameter(torch.rand(params.n_in, params.decoder_span, params.rf_size, params.rf_size), requires_grad=True)
            self._latency_mask = nn.Parameter(self._build_latency_mask(params.latency, self._encoder_weight.shape), requires_grad=False)

            mem_beta = np.exp(-params.dt / params.mem_tc)
            self._neurons = SNN(params.n_in, mem_beta, params.recurrent_type, params.frac_inhibitory, params.autapses, decrease_gabba)

            # Initialise weights
            k_encoder = params.encoder_span * params.rf_size * params.rf_size
            self.init_weight(self._encoder_weight, "uniform", a=-1 / np.sqrt(k_encoder), b=1 / np.sqrt(k_encoder))
            self.init_weight(self._encoder_bias, "constant", val=0.2)

            k_decoder = params.n_in * params.decoder_span
            self.init_weight(self._decoder_weight, "uniform", a=-1 / np.sqrt(k_decoder), b=1 / np.sqrt(k_decoder))
            self.init_weight(self._decoder_bias, "constant", val=0)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "params": self.params.hyperparams}

    @property
    def encoder_weight(self):
        return self._latency_mask * self._encoder_weight

    def forward(self, x, mode="train", stride=1, current=None):
        # x: b x n x t x h x w
        if mode in ["train", "val"]:
            assert x.shape[-1] == x.shape[-2]
            assert x.shape[-1] == self.params.rf_size

        # Add noise to input (photo receptors are noisy)
        x = x + torch.normal(0, self.params.photo_noise, size=x.shape).to(x.device)

        # Compute input current
        input_current = F.conv3d(x, self.encoder_weight, self._encoder_bias, stride=(1, stride, stride))
        abs_input_current = F.conv3d(x.abs(), self.encoder_weight.abs(), self._encoder_bias.abs(), stride=(1, stride, stride))

        # Add noise to input current
        noise = self.params.neural_noise * torch.normal(0, 1, size=input_current.shape).to(x.device)
        if self.params.noise_type == "+":
            input_current = input_current + noise
        elif self.params.noise_type == "*":
            input_current = input_current * (1 + noise)

        if current is not None:
            input_current += current

        # Obtain neuron outputs
        if mode == "just_spikes":
            # This will return spikes over all spatial locations
            if x.shape[-1] == 20:
                return self._neurons(input_current[:, :, :, 0, 0], mode)[0]
            else:
                return self._neurons(input_current, mode)

        neuron_outputs = self._neurons(input_current[:, :, :, 0, 0], mode)
        spikes = neuron_outputs[0]

        if mode == "ex_in_decode":
            ex_output = self._spikes_to_predicted_clip(spikes[:, self._neurons.excitatory_idx], self._decoder_weight[self._neurons.excitatory_idx])
            in_output = self._spikes_to_predicted_clip(spikes[:, self._neurons.inhibitory_idx], self._decoder_weight[self._neurons.inhibitory_idx])

            return ex_output, in_output

        # Decode neuron spikes to output frame
        output = self._spikes_to_predicted_clip(spikes[:, self._neurons.excitatory_idx], self._decoder_weight[self._neurons.excitatory_idx])
        output = output + self._spikes_to_predicted_clip(spikes[:, self._neurons.inhibitory_idx], self._decoder_weight[self._neurons.inhibitory_idx])
        ex_abs_output = self._spikes_to_predicted_clip(spikes[:, self._neurons.excitatory_idx], self._decoder_weight.abs()[self._neurons.excitatory_idx])
        in_abs_output = self._spikes_to_predicted_clip(spikes[:, self._neurons.inhibitory_idx], self._decoder_weight.abs()[self._neurons.inhibitory_idx])

        if mode in ["train"]:
            abs_in_graded_current = abs_input_current[:, self._neurons.inhibitory_idx].mean()
            abs_ex_graded_current = abs_input_current[:, self._neurons.excitatory_idx].mean()

            excitatory_current_to_each_unit = neuron_outputs[1]
            inhibitory_current_to_each_unit = neuron_outputs[2]
            abs_recurrent_current_to_inhibitory_neurons = excitatory_current_to_each_unit[:, self._neurons.inhibitory_idx].mean() + inhibitory_current_to_each_unit[:, self._neurons.inhibitory_idx].abs().mean()
            abs_recurrent_current_to_excitatory_neurons = excitatory_current_to_each_unit[:, self._neurons.excitatory_idx].mean() + inhibitory_current_to_each_unit[:, self._neurons.excitatory_idx].abs().mean()

            return output, abs_in_graded_current, abs_ex_graded_current, abs_recurrent_current_to_excitatory_neurons, abs_recurrent_current_to_inhibitory_neurons, ex_abs_output.mean(), in_abs_output.mean(), spikes
        elif mode == "val":
            mem = neuron_outputs[1]
            ex_rec_current = neuron_outputs[2]
            in_rec_current = neuron_outputs[3]

            return output, spikes, mem, ex_rec_current, in_rec_current, input_current

    def _spikes_to_predicted_clip(self, spikes, decoder_weight):
        _, _, sim_length = spikes.shape
        output_list = []

        for t in range(sim_length - self.params.decoder_span):
            output = torch.einsum("bnt, nthw -> bhw", spikes[:, :, t:t+self.params.decoder_span], decoder_weight) + self._decoder_bias
            output_list.append(output)

        return torch.stack(output_list, dim=1).unsqueeze(1)

    def _build_latency_mask(self, conduction_latency, shape):
        mask = torch.ones(shape)
        enc_len = shape[2]
        assert conduction_latency < enc_len
        mask[:, :, enc_len - conduction_latency:] = 0

        return mask
