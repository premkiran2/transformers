from transformers.configuration_utils import PretrainedConfig

class CodeT5pConfig(PretrainedConfig):
    model_type = "codet5p"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, **kwargs):
        # Provide default encoder and decoder configs if not specified
        if "encoder" not in kwargs:
            kwargs["encoder"] = {
                "num_layers": kwargs.get("num_encoder_layers", 12),
                "d_model": kwargs.get("d_model", 768),
                "num_attention_heads": kwargs.get("num_attention_heads", 12),
                "d_ff": kwargs.get("d_ff", 3072),
                "dropout_rate": kwargs.get("dropout_rate", 0.1)
            }
        if "decoder" not in kwargs:
            kwargs["decoder"] = {
                "num_layers": kwargs.get("num_decoder_layers", 12),
                "d_model": kwargs.get("d_model", 768),
                "num_attention_heads": kwargs.get("num_attention_heads", 12),
                "d_ff": kwargs.get("d_ff", 3072),
                "dropout_rate": kwargs.get("dropout_rate", 0.1)
            }
        super().__init__(**kwargs)
        self.encoder = kwargs["encoder"]
        self.decoder = kwargs["decoder"]

    def to_dict(self):
        """Ensure encoder/decoder configs are included in the config dict."""
        output = super().to_dict()
        output["encoder"] = self.encoder
        output["decoder"] = self.decoder
        return output