from typing import Any, NamedTuple

import modal
import pandas as pd
import requests
import torch
from torch import Tensor

MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_0/width_16k/canonical"


class SteeringConfig(NamedTuple):
    steering_feature: int
    max_act: float
    steering_strength: float


app = modal.App("smol-interp")

image = modal.Image.debian_slim().pip_install("transformer-lens", "sae-lens")

volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": volume},
    scaledown_window=300,
)
class ModalInterp:
    model: Any = None  # type: ignore
    sae: Any = None  # type: ignore
    cfg_dict: Any = None  # type: ignore
    sparsity: Any = None  # type: ignore
    explanations_df: Any = None  # type: ignore

    @modal.enter()
    def load(self) -> None:
        import os
        import pickle

        import torch
        from sae_lens import (
            SAE,
            HookedSAETransformer,
        )

        self.model = HookedSAETransformer.from_pretrained_no_processing(
            MODEL_NAME,
            device="cuda",
            torch_dtype=torch.bfloat16,
            cache_dir="/cache",
        )

        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=SAE_RELEASE, sae_id=SAE_ID, device="cuda", force_download=False
        )

        cache_path = "/cache/neuronpedia_explanations.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.explanations_df = pickle.load(f)
        else:
            self.explanations_df = self._fetch_neuronpedia_explanations()
            with open(cache_path, "wb") as f:
                pickle.dump(self.explanations_df, f)

    def _fetch_neuronpedia_explanations(
        self, sae_release: str = SAE_RELEASE, sae_id: str = SAE_ID
    ) -> pd.DataFrame:
        from sae_lens.loading.pretrained_saes_directory import (
            get_pretrained_saes_directory,
        )

        release = get_pretrained_saes_directory()[sae_release]
        neuronpedia_id = release.neuronpedia_id[sae_id]

        url = "https://www.neuronpedia.org/api/explanation/export?modelId={}&saeId={}".format(
            *neuronpedia_id.split("/")
        )
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
        df = pd.DataFrame(response.json())
        df["index"] = df["index"].astype(int)
        return df

    @modal.method()
    def get_activated_features_in_input(
        self, prompt: str, top_k: int = 5
    ) -> tuple[list[str], list[list[tuple[float, int]]]]:
        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        logits, cache = self.model.run_with_cache_with_saes(prompt, saes=[self.sae])

        assert logits is not None and isinstance(logits, torch.Tensor)

        tokens_raw = self.model.to_str_tokens(prompt)
        tokens: list[str] = (
            tokens_raw if isinstance(tokens_raw, list) else [str(tokens_raw)]
        )

        cachename = f"{self.sae.cfg.metadata.hook_name}.hook_sae_acts_post"
        sae_acts_post: Tensor = cache[cachename][0]

        features_per_token: list[list[tuple[float, int]]] = []
        for pos in range(sae_acts_post.shape[0]):
            top_acts, top_indices = sae_acts_post[pos].topk(top_k)
            features_per_token.append(
                [
                    (float(act.item()), int(idx.item()))
                    for act, idx in zip(top_acts, top_indices)
                ]
            )

        return (tokens, features_per_token)

    def _generate_and_capture_features(
        self, prompt: str, max_new_tokens: int = 20, top_k: int = 5
    ) -> tuple[str, list[str], list[list[tuple[float, int]]]]:
        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        captured_features: list[list[tuple[float, int]]] = []

        def capture_hook(activations: Tensor, hook: Any) -> Tensor:
            feature_acts = self.sae.encode(activations)
            last_token_acts: Tensor = feature_acts[0, -1, :]

            top_acts, top_indices = last_token_acts.topk(top_k)
            captured_features.append(
                [
                    (float(act.item()), int(idx.item()))
                    for act, idx in zip(top_acts, top_indices)
                ]
            )

            return activations

        input_ids = self.model.to_tokens(
            prompt, prepend_bos=self.sae.cfg.metadata.prepend_bos
        )

        with self.model.hooks(
            fwd_hooks=[(self.sae.cfg.metadata.hook_name, capture_hook)]
        ):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=True,
                prepend_bos=self.sae.cfg.metadata.prepend_bos,
            )

        full_text: str = self.model.tokenizer.decode(
            output[0], skip_special_tokens=True
        )

        prompt_length = input_ids.shape[1]
        generated_token_ids: Tensor = output[0, prompt_length:]
        generated_tokens: list[str] = [
            str(self.model.to_string(token_id)) for token_id in generated_token_ids
        ]

        return (full_text, generated_tokens, captured_features)

    @modal.method()
    def get_activated_features_in_output(
        self, prompt: str, max_new_tokens: int = 20, top_k: int = 5
    ) -> tuple[str, list[str], list[list[tuple[float, int]]]]:
        return self._generate_and_capture_features(prompt, max_new_tokens, top_k)

    @modal.method()
    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "n_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
        }

    @modal.method()
    def get_explanations_df(self) -> pd.DataFrame:
        return self.explanations_df

    @modal.method()
    def get_autosteer(self, _prompt: str) -> list[SteeringConfig] | None:
        return None

    @modal.method()
    def generate_with_explanations(
        self, prompt: str, max_new_tokens: int = 20, top_k: int = 5
    ) -> dict[str, Any]:
        full_text, generated_tokens, output_features = (
            self._generate_and_capture_features(prompt, max_new_tokens, top_k)
        )

        token_explanations = []
        for token, features in zip(generated_tokens, output_features):
            feature_explanations = []
            for activation, feature_idx in features:
                explanation_row = self.explanations_df[
                    self.explanations_df["index"] == feature_idx
                ]

                if not explanation_row.empty:
                    explanation = explanation_row.iloc[0].get(
                        "description", "No description available"
                    )
                else:
                    explanation = f"No explanation found for feature {feature_idx}"

                feature_explanations.append(
                    {
                        "feature_idx": feature_idx,
                        "activation": activation,
                        "explanation": explanation,
                    }
                )

            token_explanations.append(
                {"token": token, "features": feature_explanations}
            )

        return {
            "full_text": full_text,
            "tokens": generated_tokens,
            "token_explanations": token_explanations,
        }

    @modal.method()
    def run_with_steering(
        self,
        prompt: str,
        steering_configs: list[SteeringConfig],
        max_new_tokens: int = 100,
    ) -> str:
        from functools import partial

        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        def steering(
            activations: Tensor,
            hook: Any,
            steering_vector: Tensor | None = None,
            max_act: float = 1.0,
            steering_strength: float = 1.0,
        ) -> Tensor:
            if steering_vector is None:
                return activations
            return activations + max_act * steering_strength * steering_vector

        input_ids = self.model.to_tokens(
            prompt, prepend_bos=self.sae.cfg.metadata.prepend_bos
        )

        combined_steering_vector: Tensor = torch.zeros(
            self.sae.cfg.d_sae, device=self.model.cfg.device
        )

        for config in steering_configs:
            steering_vector: Tensor = self.sae.W_dec[config.steering_feature].to(
                self.model.cfg.device
            )
            combined_steering_vector += (
                config.max_act * config.steering_strength * steering_vector
            )

        steering_hook = partial(
            steering,
            steering_vector=combined_steering_vector,
            max_act=1.0,
            steering_strength=1.0,
        )

        with self.model.hooks(
            fwd_hooks=[(self.sae.cfg.metadata.hook_name, steering_hook)]
        ):
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=True,
                prepend_bos=self.sae.cfg.metadata.prepend_bos,
            )

        return str(self.model.tokenizer.decode(output[0]))


@app.local_entrypoint()
def main() -> None:
    model = ModalInterp()  # type: ignore

    info = model.get_model_info.remote()
    print(f"Model info: {info}")

    prompt = "The capital of France is"
    result = model.generate_with_explanations.remote(prompt, max_new_tokens=3, top_k=5)
    print(f"Full text: {result['full_text']}")
    for i, token_data in enumerate(result["token_explanations"]):
        print(f"\n[{i}] Token: '{token_data['token']}'")
        for feat in token_data["features"]:
            print(
                f"Feature {feat['feature_idx']} (activation: {feat['activation']:.2f})"
            )
            print(f"{feat['explanation']}")
