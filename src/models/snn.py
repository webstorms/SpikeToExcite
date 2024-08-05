import torch
import torch.nn as nn
import numpy as np
import devtorch


class FastSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale=100):
        ctx.scale = scale
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0

        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.scale * torch.abs(input) + 1.0) ** 2

        return grad, None


class SNN(devtorch.DevModel):

    def __init__(self, n_out, mem_beta, recurrent_type, frac_inhibitory, autapses, decrease_gabba):
        super().__init__()
        self._autapses = autapses
        self._decrease_gabba = decrease_gabba

        # Membrane
        self._beta = nn.Parameter(data=torch.clamp(torch.normal(mem_beta, 0.01, (n_out,)), 0, 1), requires_grad=True)

        # Recurrent
        self._recurrent_weight = nn.Parameter(torch.rand(n_out, n_out), requires_grad=True)
        self._neuron_type_weight = nn.Parameter(torch.normal(0, 1, size=(n_out,)), requires_grad=recurrent_type == "learn_dale")
        self._autapse_mask = nn.Parameter(self._build_autapse_mask(n_out), requires_grad=False)
        self._in_mask = nn.Parameter(torch.ones(n_out, n_out), requires_grad=False)

        # Initialise weights
        self.init_weight(self._recurrent_weight, "uniform", a=-1 / np.sqrt(n_out), b=1 / np.sqrt(n_out))
        n_inhibitory_neurons = int(frac_inhibitory * n_out)
        self.init_weight(self._neuron_type_weight, "constant", val=0.1)  # Fix all neurons to be ex
        self._neuron_type_weight.data[:n_inhibitory_neurons] = -0.1  # Fix these neurons to be inhib
        self._in_mask.data[:, :n_inhibitory_neurons] *= (1 - self._decrease_gabba)

    @property
    def inhibitory_idx(self):
        return self._neuron_type_weight < 0

    @property
    def excitatory_idx(self):
        return self._neuron_type_weight > 0

    @property
    def beta(self):
        return torch.clamp(self._beta, min=0.001, max=0.999)

    @property
    def recurrent_weight(self):
        abs_recurrent_weight = torch.clamp(torch.abs(self._recurrent_weight), min=1e-8, max=1e5)
        dale_recurrent_weight = torch.einsum("ij, j -> ij", abs_recurrent_weight, self._neuron_type_weight)
        gabba_decrease_weight = torch.einsum("ij, ij -> ij", dale_recurrent_weight, self._in_mask)

        if self._autapses:
            return gabba_decrease_weight
        else:
            return gabba_decrease_weight * self._autapse_mask

    def get_recurrent_current(self, spikes):
        ex_neuron_idxs = self.excitatory_idx
        in_neuron_idxs = self.inhibitory_idx
        ex_input = torch.einsum("ij, bj... -> bi...", self.recurrent_weight[:, ex_neuron_idxs], spikes[:, ex_neuron_idxs])
        in_input = torch.einsum("ij, bj... -> bi...", self.recurrent_weight[:, in_neuron_idxs], spikes[:, in_neuron_idxs])

        return ex_input, in_input

    def forward(self, x, mode="train"):
        # x: b x n x t

        mem_list = []
        spike_list = []
        ex_currents_list = []
        in_currents_list = []

        spikes = torch.zeros_like(x).to(x.device)[:, :, 0]
        mem = torch.zeros_like(x).to(x.device)[:, :, 0]

        for t in range(x.shape[2]):
            input_current = x[:, :, t]

            # Get recurrent currents
            ex_current, in_current = self.get_recurrent_current(spikes.detach())
            ex_currents_list.append(ex_current)
            in_currents_list.append(in_current)
            input_current = input_current + (ex_current + in_current)

            # Update membrane potentials
            new_mem = torch.einsum("bn..., n -> bn...", mem, self.beta) + input_current

            # Output spikes
            spikes = FastSigmoid.apply(new_mem - 1)

            # ===> Mod to original model <===
            new_mem = torch.clamp(new_mem, -2, 1)  # Can't be more negative than -2

            mem = new_mem * (1 - spikes.detach())
            spike_list.append(spikes)

            # Validation mode variables
            mem_list.append(new_mem)

        if mode in ["train"]:
            return torch.stack(spike_list, dim=2), torch.stack(ex_currents_list, dim=2), torch.stack(in_currents_list, dim=2)
        elif mode == "val":
            return torch.stack(spike_list, dim=2), torch.stack(mem_list, dim=2), torch.stack(ex_currents_list, dim=2), torch.stack(in_currents_list, dim=2)
        elif mode in ["just_spikes", "ex_in_decode"]:
            return torch.stack(spike_list, dim=2), torch.stack(mem_list, dim=2)

    def _build_autapse_mask(self, n_out):
        no_self_connection_mask = torch.ones(n_out, n_out)
        for i in range(n_out):
            no_self_connection_mask[i, i] = 0

        return no_self_connection_mask
