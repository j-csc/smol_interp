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

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "torch",
    "transformer-lens",
    "sae-lens",
    "pandas",
    "requests",
)

volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": volume},
    scaledown_window=300,
)
class ModalInterp:
    model: Any = None
    sae: Any = None
    cfg_dict: Any = None
    sparsity: Any = None
    explanations_df: Any = None
    explanations_by_index: Any = None

    @modal.enter()
    def load(self) -> None:
        import os
        import pickle

        from sae_lens import SAE, HookedSAETransformer

        self.model = HookedSAETransformer.from_pretrained_no_processing(
            MODEL_NAME,
            device="cuda",
            torch_dtype=torch.float16,
            cache_dir="/cache",
        )

        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=SAE_ID,
            device="cuda",
            force_download=False,
        )

        cache_path = "/cache/neuronpedia_explanations.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.explanations_df = pickle.load(f)
        else:
            self.explanations_df = self._fetch_neuronpedia_explanations()
            with open(cache_path, "wb") as f:
                pickle.dump(self.explanations_df, f)
            volume.commit()

        if (
            isinstance(self.explanations_df, pd.DataFrame)
            and "index" in self.explanations_df.columns
        ):
            idx = self.explanations_df["index"].astype(int)
            desc = (
                self.explanations_df["description"]
                if "description" in self.explanations_df.columns
                else pd.Series([""] * len(self.explanations_df))
            )
            self.explanations_by_index = (
                pd.DataFrame({"index": idx, "description": desc})
                .dropna(subset=["index"])
                .drop_duplicates(subset=["index"], keep="first")
                .set_index("index")["description"]
                .to_dict()
            )
        else:
            self.explanations_by_index = {}

    def _fetch_neuronpedia_explanations(
        self, sae_release: str = SAE_RELEASE, sae_id: str = SAE_ID
    ) -> pd.DataFrame:
        from sae_lens.loading.pretrained_saes_directory import (
            get_pretrained_saes_directory,
        )

        release = get_pretrained_saes_directory()[sae_release]
        neuronpedia_id = release.neuronpedia_id[sae_id]

        model_id, sae_np_id = neuronpedia_id.split("/", 1)
        url = (
            "https://www.neuronpedia.org/api/explanation/export"
            f"?modelId={model_id}&saeId={sae_np_id}"
        )

        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data)
        if "index" in df.columns:
            df["index"] = df["index"].astype(int, errors="ignore")
        return df

    def _get_input_features(
        self, prompt: str, top_k: int = 5
    ) -> tuple[list[str], list[list[tuple[float, int]]]]:
        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        _, cache = self.model.run_with_cache_with_saes(prompt, saes=[self.sae])

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
                    (float(a.item()), int(i.item()))
                    for a, i in zip(top_acts, top_indices)
                ]
            )

        return tokens, features_per_token

    def _generate_and_capture_features(
        self, prompt: str, max_new_tokens: int = 20, top_k: int = 5
    ) -> tuple[str, list[str], list[list[tuple[float, int]]]]:
        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        captured_features: list[list[tuple[float, int]]] = []
        seen_prompt_len: int | None = None

        def capture_hook(activations: Tensor, hook: Any) -> Tensor:
            nonlocal seen_prompt_len
            seq_len = int(activations.shape[1])
            if seen_prompt_len is None:
                seen_prompt_len = seq_len
                return activations
            if seq_len == seen_prompt_len:
                return activations

            feature_acts = self.sae.encode(activations)
            last_token_acts: Tensor = feature_acts[0, -1, :]

            top_acts, top_indices = last_token_acts.topk(top_k)
            captured_features.append(
                [
                    (float(a.item()), int(i.item()))
                    for a, i in zip(top_acts, top_indices)
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

        prompt_length = int(input_ids.shape[1])
        generated_token_ids: Tensor = output[0, prompt_length:]
        generated_tokens: list[str] = [
            self.model.tokenizer.decode([int(t.item())], skip_special_tokens=False)
            for t in generated_token_ids
        ]

        if len(captured_features) > len(generated_tokens):
            captured_features = captured_features[-len(generated_tokens) :]
        elif len(captured_features) < len(generated_tokens):
            pad = [[] for _ in range(len(generated_tokens) - len(captured_features))]
            captured_features = pad + captured_features

        return full_text, generated_tokens, captured_features

    def _explain_features(
        self, token_features: list[list[tuple[float, int]]]
    ) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        for feats in token_features:
            rows: list[dict[str, Any]] = []
            for activation, feature_idx in feats:
                explanation = self.explanations_by_index.get(
                    feature_idx, f"No explanation found for feature {feature_idx}"
                )
                if not explanation:
                    explanation = "No description available"
                rows.append(
                    {
                        "feature_idx": int(feature_idx),
                        "activation": float(activation),
                        "explanation": explanation,
                    }
                )
            out.append(rows)
        return out

    def generate_with_explanations(
        self, prompt: str, max_new_tokens: int = 20, top_k: int = 5
    ) -> dict[str, Any]:
        input_tokens, input_features = self._get_input_features(prompt, top_k)
        input_rows = self._explain_features(input_features)
        input_token_explanations = [
            {"token": tok, "features": feats}
            for tok, feats in zip(input_tokens, input_rows)
        ]

        full_text, generated_tokens, output_features = (
            self._generate_and_capture_features(prompt, max_new_tokens, top_k)
        )
        output_rows = self._explain_features(output_features)
        output_token_explanations = [
            {"token": tok, "features": feats}
            for tok, feats in zip(generated_tokens, output_rows)
        ]

        feature_max_activations: dict[int, float] = {}
        for token_data in input_token_explanations:
            for feat in token_data["features"]:
                idx = int(feat["feature_idx"])
                act = float(feat["activation"])
                prev = feature_max_activations.get(idx)
                if prev is None or act > prev:
                    feature_max_activations[idx] = act

        for token_data in output_token_explanations:
            for feat in token_data["features"]:
                idx = int(feat["feature_idx"])
                act = float(feat["activation"])
                prev = feature_max_activations.get(idx)
                if prev is None or act > prev:
                    feature_max_activations[idx] = act

        return {
            "prompt": prompt,
            "full_text": full_text,
            "input_token_explanations": input_token_explanations,
            "output_token_explanations": output_token_explanations,
            "feature_max_activations": feature_max_activations,
        }

    def run_with_steering(
        self,
        prompt: str,
        steering_configs: list[SteeringConfig],
        max_new_tokens: int = 100,
    ) -> str:
        from functools import partial

        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        input_ids = self.model.to_tokens(
            prompt, prepend_bos=self.sae.cfg.metadata.prepend_bos
        )

        combined_steering_vector: Tensor = torch.zeros(
            self.sae.cfg.d_in, device=self.model.cfg.device
        )

        for cfg in steering_configs:
            combined_steering_vector += (
                cfg.max_act
                * cfg.steering_strength
                * self.sae.W_dec[cfg.steering_feature].to(self.model.cfg.device)
            )

        def steering(activations: Tensor, hook: Any, steering_vector: Tensor) -> Tensor:
            v = steering_vector.to(device=activations.device, dtype=activations.dtype)
            acts = activations.clone()
            acts[:, -1, :] = acts[:, -1, :] + v
            return acts

        steering_hook = partial(steering, steering_vector=combined_steering_vector)

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

        return self.model.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "n_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
        }

    def get_explanations(self) -> list[dict[str, Any]]:
        if isinstance(self.explanations_df, pd.DataFrame):
            return self.explanations_df.to_dict(orient="records")
        return []

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from pydantic import BaseModel, Field

        class GenerateRequest(BaseModel):
            prompt: str
            max_new_tokens: int = Field(default=20, ge=0, le=512)
            top_k: int = Field(default=5, ge=1, le=100)

        class SteeringItem(BaseModel):
            steering_feature: int
            max_act: float
            steering_strength: float

        class SteerRequest(BaseModel):
            prompt: str
            steering_configs: list[SteeringItem] = Field(default_factory=list)
            max_new_tokens: int = Field(default=100, ge=0, le=512)

        api = FastAPI()

        @api.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @api.get("/info")
        def info() -> dict[str, Any]:
            return self.get_model_info()

        @api.get("/explanations")
        def explanations() -> list[dict[str, Any]]:
            return self.get_explanations()

        @api.post("/generate")
        def generate(req: GenerateRequest) -> dict[str, Any]:
            return self.generate_with_explanations(
                req.prompt, max_new_tokens=req.max_new_tokens, top_k=req.top_k
            )

        @api.post("/steer")
        def steer(req: SteerRequest) -> dict[str, Any]:
            cfgs = [
                SteeringConfig(
                    steering_feature=int(c.steering_feature),
                    max_act=float(c.max_act),
                    steering_strength=float(c.steering_strength),
                )
                for c in req.steering_configs
            ]
            text = self.run_with_steering(
                req.prompt, cfgs, max_new_tokens=req.max_new_tokens
            )
            return {"prompt": req.prompt, "full_text": text}

        return api
