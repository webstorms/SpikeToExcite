
class V1Parameters:

    def __init__(self, n_in, rf_size, encoder_span, decoder_span, latency, mem_tc, recurrent_type, autapses, frac_inhibitory, dt, photo_noise=0, neural_noise=0, noise_type="*"):
        assert recurrent_type in ["fixed_dale", "learn_dale"]
        assert noise_type in ["+", "*"]
        self.n_in = n_in
        self.rf_size = rf_size
        self.encoder_span = encoder_span
        self.decoder_span = decoder_span
        self.latency = latency
        self.mem_tc = mem_tc
        self.recurrent_type = recurrent_type
        self.autapses = autapses
        self.frac_inhibitory = frac_inhibitory
        self.dt = dt
        self.photo_noise = photo_noise
        self.neural_noise = neural_noise
        self.noise_type = noise_type

    @property
    def hyperparams(self):
        return {"n_in": self.n_in, "rf_size": self.rf_size, "encoder_span": self.encoder_span, "decoder_span": self.decoder_span, "latency": self.latency, "mem_tc": self.mem_tc, "recurrent_type": self.recurrent_type, "autapses": self.autapses, "frac_inhibitory": self.frac_inhibitory, "dt": self.dt, "photo_noise": self.photo_noise, "neural_noise": self.neural_noise, "noise_type": self.noise_type}