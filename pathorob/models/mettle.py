import torch
import torchvision.transforms as transforms

from pathorob.models.utils import ModelWrapper


class Mettle15ModelWrapper(ModelWrapper):
    """
    METTLE 1.5 (Slideflow Labs): https://huggingface.co/slideflow-labs/mettle-1.5

    License: CC-BY-NC-ND-4.0. Access is gated with automatic approval: accept the terms on the
    model page, then pass a Hugging Face token via `--model_args '{"hf_token": "<TOKEN>"}'`
    (or log in beforehand with `huggingface-cli login`).
    Requires: timm>=1.0.15, transformers>=4.48, safetensors>=0.4.
    """

    def __init__(self, feature_view="cls", hf_token=None):
        """
        :param hf_token: Optional Hugging Face token with access to the gated model repository.
        :param feature_view: "cls" for the 1536-dim refined CLS embedding (model default), or
            "cls_mean" for the 3072-dim refined CLS + mean spatial patch token embedding.
        """
        from transformers import AutoModel
        if hf_token is not None:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        if feature_view not in ("cls", "cls_mean"):
            raise ValueError(f"feature_view must be 'cls' or 'cls_mean', got '{feature_view}'.")
        self.feature_view = feature_view
        self.model = AutoModel.from_pretrained("slideflow-labs/mettle-1.5", trust_remote_code=True)
        self.model.eval()

    def get_model(self):
        return self.model

    def get_preprocess(self):
        # As specified in the model's preprocessor_config.json
        return transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    def extract(self, data):
        return self.model(pixel_values=data, feature_view=self.feature_view).pooler_output
