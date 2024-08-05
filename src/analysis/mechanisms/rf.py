import torch
from brainbox import rfs
from brainbox.neural.correlation import cc


from src.models import load_model


class RFAnalyses:

    def __init__(self, root, model_id, decrease_gabba=0.8):
        self._healthy_model = load_model(f"{root}/data", model_id, 0.0)
        self._seizure_model = load_model(f"{root}/data", model_id, decrease_gabba)
        self._healthy_model_rf_query = _RFQuery(self._healthy_model)
        self._seizure_model_rf_query = _RFQuery(self._seizure_model)
    
    def get_mean_power_per_spatial_rf(self):
        healthy_model_rf_power = (self._healthy_model_rf_query.all_rfs**2).mean((1, 2))
        seizure_model_rf_power = (self._seizure_model_rf_query.all_rfs**2).mean((1, 2))

        return healthy_model_rf_power, seizure_model_rf_power

    def get_rfs(self, idxs):
        healthy_model_rfs = torch.stack([self._healthy_model_rf_query.all_rfs[i] for i in idxs])
        seizure_model_rfs = torch.stack([self._healthy_model_rf_query.all_rfs[i] for i in idxs])

        return healthy_model_rfs, seizure_model_rfs

    def get_cc_between_models_rfs(self, idxs):
        rf_ccs = cc(self._healthy_model_rf_query.all_rfs.flatten(1, 2), self._seizure_model_rf_query.all_rfs.flatten(1, 2))

        return rf_ccs if idxs is None else rf_ccs[idxs]


class _RFQuery:

    def __init__(self, model):
        torch.manual_seed(42)
        self.model = model
        self.all_strfs = self._build_strfs(self.model, samples=200)
        self.all_rfs = self._get_all_highest_power_spatial_rf(self.all_strfs)

    # Building and querying STA RFs

    def _build_strfs(self, model, rf_len=18, t_len=100, noise_var=10, samples=2000, batch_size=50, rf_h=20, rf_w=20, device="cuda", **kwargs):

        def model_output(noise):
            with torch.no_grad():
                return model(noise.unsqueeze(1), mode="val", **kwargs)[1]

        return rfs.sta(model_output, rf_len, rf_h, rf_w, t_len, noise_var, samples, batch_size, device)

    def _get_all_highest_power_spatial_rf(self, spatiotemporal_rfs):
        # spatiotemporal_rfs: n_units, rf_len, rf_shape, rf_shape
        rfs = []

        for i in range(len(spatiotemporal_rfs)):
            spatial_rf = self._get_highest_power_spatial_rf(spatiotemporal_rfs[i].detach().cpu().float())
            rfs.append(spatial_rf)
        rfs = torch.stack(rfs)

        return rfs

    def _get_highest_power_spatial_rf(self, spatiotemporal_rf):
        # spatiotemporal_rf: rf_len, rf_shape, rf_shape
        power_at_timesteps = torch.pow(spatiotemporal_rf, 2).mean(dim=(1, 2))
        t = power_at_timesteps.argmax().item()
        spatial_rf = spatiotemporal_rf[t]

        return spatial_rf

    # Query functions

    def get_rfs(self):
        model_one_subfield = self.all_rfs[224] / self.all_rfs.max()
        model_two_subfield = self.all_rfs[309] / self.all_rfs.max()
        model_three_subfield = self.all_rfs[531] / self.all_rfs.max()

        return [model_one_subfield, model_two_subfield, model_three_subfield]
