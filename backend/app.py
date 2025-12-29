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


app = modal.App("autosteer")

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]",
    "torch",
    "transformer-lens",
    "sae-lens",
    "pandas",
    "requests",
    "openai",
)

volume = modal.Volume.from_name("model-cache", create_if_missing=True)


@app.cls(
    gpu="T4",
    image=image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("openai-secret"),
    ],
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
    openai_client: Any = None
    feature_embeddings: Any = None
    feature_indices: Any = None

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

        from openai import OpenAI

        self.openai_client = OpenAI()

        embeddings_path = "/cache/openai_embeddings.pkl"
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                cache_data = pickle.load(f)
                self.feature_embeddings = cache_data["embeddings"]
                self.feature_indices = cache_data["indices"]
        else:
            valid_features = [
                (idx, desc)
                for idx, desc in self.explanations_by_index.items()
                if desc and isinstance(desc, str) and len(desc.strip()) > 0
            ]

            if valid_features:
                self.feature_indices = [idx for idx, _ in valid_features]
                descriptions = [desc for _, desc in valid_features]

                print(f"Computing embeddings for {len(descriptions)} features...")
                embeddings = []
                batch_size = 2048

                for i in range(0, len(descriptions), batch_size):
                    batch = descriptions[i : i + batch_size]
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-ada-002", input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    print(
                        f"  Processed {min(i + batch_size, len(descriptions))}/{len(descriptions)}"
                    )

                import numpy as np

                self.feature_embeddings = np.array(embeddings, dtype=np.float32)

                with open(embeddings_path, "wb") as f:
                    pickle.dump(
                        {
                            "embeddings": self.feature_embeddings,
                            "indices": self.feature_indices,
                        },
                        f,
                    )
                volume.commit()
                print("Embeddings cached successfully")
            else:
                self.feature_indices = []
                self.feature_embeddings = None

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

    def _compute_feature_dla(
        self,
        token_residual: Tensor,
        feature_acts: Tensor,
        target_token_id: int | None = None,
        top_k: int = 5,
    ) -> list[tuple[float, int]] | Tensor:
        W_dec = self.sae.W_dec.to(token_residual.device, token_residual.dtype)
        W_U = self.model.W_U.to(token_residual.device, token_residual.dtype)

        feature_to_logits = W_dec @ W_U

        if target_token_id is not None:
            contributions = feature_acts * feature_to_logits[:, target_token_id]
            top_vals, top_idx = contributions.topk(top_k)
            return [(float(v.item()), int(i.item())) for v, i in zip(top_vals, top_idx)]

        logit_contributions = feature_acts @ feature_to_logits
        return logit_contributions

    def _find_max_activating_examples(
        self,
        feature_idx: int,
        texts: list[str],
        n_examples: int = 5,
    ) -> list[dict[str, Any]]:
        from sae_lens import SAE

        assert self.sae is not None and isinstance(self.sae, SAE)

        examples = []
        for text in texts:
            _, cache = self.model.run_with_cache_with_saes(text, saes=[self.sae])
            cachename = f"{self.sae.cfg.metadata.hook_name}.hook_sae_acts_post"
            sae_acts_post: Tensor = cache[cachename][0]

            tokens = self.model.to_str_tokens(text)
            if not isinstance(tokens, list):
                tokens = [str(tokens)]

            for pos in range(sae_acts_post.shape[0]):
                activation = float(sae_acts_post[pos, feature_idx].item())
                if activation > 0:
                    examples.append(
                        {
                            "text": text,
                            "token": tokens[pos] if pos < len(tokens) else "",
                            "position": pos,
                            "activation": activation,
                        }
                    )

        examples.sort(key=lambda x: x["activation"], reverse=True)
        return examples[:n_examples]

    def _search_features_by_description(
        self, query: str, top_k: int = 5
    ) -> list[tuple[int, float, str]]:
        import numpy as np

        if self.feature_embeddings is None or len(self.feature_indices) == 0:
            return []

        query_response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002", input=[query]
        )
        query_embedding = np.array(query_response.data[0].embedding, dtype=np.float32)

        similarities = np.dot(self.feature_embeddings, query_embedding) / (
            np.linalg.norm(self.feature_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            feature_idx = self.feature_indices[idx]
            similarity = float(similarities[idx])
            description = self.explanations_by_index.get(feature_idx, "")
            results.append((feature_idx, similarity, description))

        return results

    def auto_interpret_feature(
        self,
        feature_idx: int,
        example_texts: list[str],
        n_examples: int = 5,
    ) -> dict[str, Any]:
        max_activating = self._find_max_activating_examples(
            feature_idx, example_texts, n_examples
        )

        if not max_activating:
            return {
                "feature_idx": feature_idx,
                "explanation": "No activations found",
                "examples": [],
            }

        prompt_parts = [
            "Below are examples of text tokens where a particular neural network feature activated strongly.",
            "Each example shows the token and its activation strength.",
            "Based on these examples, provide a concise 1-2 sentence explanation of what semantic pattern this feature detects.\n",
        ]

        for i, ex in enumerate(max_activating, 1):
            prompt_parts.append(
                f"{i}. Token: '{ex['token']}' (activation: {ex['activation']:.2f})"
            )
            prompt_parts.append(f"   Context: {ex['text']}\n")

        prompt_parts.append(
            "\nWhat pattern does this feature detect? Be specific and concise."
        )
        interpretation_prompt = "\n".join(prompt_parts)

        input_ids = self.model.to_tokens(interpretation_prompt)
        output = self.model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            stop_at_eos=True,
        )

        explanation = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
        explanation = explanation[len(interpretation_prompt) :].strip()

        return {
            "feature_idx": feature_idx,
            "explanation": explanation,
            "examples": max_activating,
        }

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

    def auto_steer(
        self,
        prompt: str,
        desired_attributes: list[str],
        max_new_tokens: int = 100,
        top_k_features: int = 3,
    ) -> dict[str, Any]:
        baseline_result = self.generate_with_explanations(
            prompt, max_new_tokens=0, top_k=10
        )
        baseline_activations = baseline_result["feature_max_activations"]

        all_features = []
        for attribute in desired_attributes:
            found_features = self._search_features_by_description(
                attribute, top_k=top_k_features
            )
            for feature_idx, relevance_score, reasoning in found_features:
                baseline_act = baseline_activations.get(feature_idx, 10.0)
                all_features.append(
                    {
                        "attribute": attribute,
                        "feature_idx": feature_idx,
                        "relevance_score": relevance_score,
                        "reasoning": reasoning,
                        "baseline_activation": baseline_act,
                    }
                )

        import json

        optimization_prompt = f"""You are an AI assistant optimizing neural network steering for text generation.

Task: Generate text with these attributes: {", ".join(desired_attributes)}
Prompt: "{prompt}"

Available features to steer:
{json.dumps(all_features, indent=2)}

For each feature, determine the optimal steering strength (typically 0.5 to 3.0). Consider:
- Relevance score (how well it matches the desired attribute)
- Baseline activation (higher baseline = feature already present, may need less boost)
- Potential interactions between features
- Risk of over-steering (too high can make text unnatural)

Respond with a JSON object:
{{
  "steering_configs": [
    {{"feature_idx": <int>, "steering_strength": <float>, "reasoning": "<brief explanation>"}},
    ...
  ],
  "overall_strategy": "<brief explanation of the steering approach>"
}}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": optimization_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        optimization_result = json.loads(response.choices[0].message.content)

        steering_configs = []
        optimized_features = []

        for config in optimization_result.get("steering_configs", []):
            feature_idx = config["feature_idx"]
            strength = config["steering_strength"]

            feature_info = next(
                (f for f in all_features if f["feature_idx"] == feature_idx), None
            )
            if not feature_info:
                continue

            max_act = feature_info["baseline_activation"]

            steering_configs.append(
                SteeringConfig(
                    steering_feature=feature_idx,
                    max_act=max_act,
                    steering_strength=strength,
                )
            )

            optimized_features.append(
                {
                    "feature_idx": feature_idx,
                    "max_act": max_act,
                    "steering_strength": strength,
                    "reasoning": config.get("reasoning", ""),
                    "attribute": feature_info["attribute"],
                    "feature_description": self.explanations_by_index.get(
                        feature_idx, ""
                    ),
                }
            )

        steered_text = self.run_with_steering(
            prompt, steering_configs, max_new_tokens=max_new_tokens
        )

        return {
            "prompt": prompt,
            "baseline_text": baseline_result["full_text"],
            "steered_text": steered_text,
            "desired_attributes": desired_attributes,
            "optimized_features": optimized_features,
            "overall_strategy": optimization_result.get("overall_strategy", ""),
        }

    def auto_steer_from_prompt(
        self,
        prompt: str,
        steering_prompt: str,
        max_new_tokens: int = 100,
        top_k_features: int = 3,
    ) -> dict[str, Any]:
        """Auto-steer based on a natural language steering prompt.

        Args:
            prompt: The text generation prompt
            steering_prompt: Natural language description of how to steer (e.g., "Make it dark and mysterious")
            max_new_tokens: Maximum tokens to generate
            top_k_features: Number of features to find per attribute

        Returns:
            Dict containing steered text, baseline, and full transparency about what was steered
        """
        import json

        attribute_extraction_prompt = f"""You are analyzing a user's request to steer text generation.

Generation prompt: "{prompt}"
Steering request: "{steering_prompt}"

Extract 2-5 specific attributes or concepts that should be amplified in the generation to satisfy the steering request.
Be specific and concrete. Focus on semantic concepts, tones, styles, or themes.

Examples:
- "Make it scary" → ["fear/dread emotions", "ominous imagery", "suspenseful pacing"]
- "More technical and formal" → ["technical terminology", "formal language", "precise descriptions"]
- "Add humor" → ["playful tone", "comedic situations", "wit/wordplay"]

Respond with a JSON object:
{{
  "attributes": ["<attribute1>", "<attribute2>", ...],
  "reasoning": "<brief explanation of why these attributes match the steering request>"
}}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": attribute_extraction_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        extraction_result = json.loads(response.choices[0].message.content)
        desired_attributes = extraction_result.get("attributes", [])
        attribute_reasoning = extraction_result.get("reasoning", "")

        baseline_result = self.generate_with_explanations(
            prompt, max_new_tokens=0, top_k=10
        )
        baseline_activations = baseline_result["feature_max_activations"]

        all_features = []
        for attribute in desired_attributes:
            found_features = self._search_features_by_description(
                attribute, top_k=top_k_features
            )
            for feature_idx, relevance_score, reasoning in found_features:
                baseline_act = baseline_activations.get(feature_idx, 10.0)
                all_features.append(
                    {
                        "attribute": attribute,
                        "feature_idx": feature_idx,
                        "relevance_score": relevance_score,
                        "reasoning": reasoning,
                        "baseline_activation": baseline_act,
                    }
                )

        optimization_prompt = f"""You are an AI assistant optimizing neural network steering for text generation.

Generation prompt: "{prompt}"
User's steering request: "{steering_prompt}"
Extracted attributes: {", ".join(desired_attributes)}

Available features to steer:
{json.dumps(all_features, indent=2)}

For each feature, determine the optimal steering strength (typically 0.5 to 3.0). Consider:
- Relevance score (how well it matches the desired attribute)
- Baseline activation (higher baseline = feature already present, may need less boost)
- Potential interactions between features
- Risk of over-steering (too high can make text unnatural)

Respond with a JSON object:
{{
  "steering_configs": [
    {{"feature_idx": <int>, "steering_strength": <float>, "reasoning": "<brief explanation>"}},
    ...
  ],
  "overall_strategy": "<brief explanation of the steering approach>"
}}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": optimization_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        optimization_result = json.loads(response.choices[0].message.content)

        steering_configs = []
        optimized_features = []

        for config in optimization_result.get("steering_configs", []):
            feature_idx = config["feature_idx"]
            strength = config["steering_strength"]

            feature_info = next(
                (f for f in all_features if f["feature_idx"] == feature_idx), None
            )
            if not feature_info:
                continue

            max_act = feature_info["baseline_activation"]

            steering_configs.append(
                SteeringConfig(
                    steering_feature=feature_idx,
                    max_act=max_act,
                    steering_strength=strength,
                )
            )

            optimized_features.append(
                {
                    "feature_idx": feature_idx,
                    "max_act": max_act,
                    "steering_strength": strength,
                    "reasoning": config.get("reasoning", ""),
                    "attribute": feature_info["attribute"],
                    "feature_description": self.explanations_by_index.get(
                        feature_idx, ""
                    ),
                }
            )

        steered_text = self.run_with_steering(
            prompt, steering_configs, max_new_tokens=max_new_tokens
        )

        return {
            "prompt": prompt,
            "steering_prompt": steering_prompt,
            "baseline_text": baseline_result["full_text"],
            "steered_text": steered_text,
            "extracted_attributes": desired_attributes,
            "attribute_reasoning": attribute_reasoning,
            "optimized_features": optimized_features,
            "overall_strategy": optimization_result.get("overall_strategy", ""),
        }

    def preview_steering(
        self,
        prompt: str,
        steering_prompt: str,
        top_k_features: int = 3,
    ) -> dict[str, Any]:
        """Preview which features would be steered without generating text."""
        import json

        attribute_extraction_prompt = f"""You are analyzing a user's request to steer text generation.

Generation prompt: "{prompt}"
Steering request: "{steering_prompt}"

Extract 2-5 specific attributes or concepts that should be amplified in the generation to satisfy the steering request.
Be specific and concrete. Focus on semantic concepts, tones, styles, or themes.

Examples:
- "Make it scary" → ["fear/dread emotions", "ominous imagery", "suspenseful pacing"]
- "More technical and formal" → ["technical terminology", "formal language", "precise descriptions"]
- "Add humor" → ["playful tone", "comedic situations", "wit/wordplay"]

Respond with a JSON object:
{{
  "attributes": ["<attribute1>", "<attribute2>", ...],
  "reasoning": "<brief explanation of why these attributes match the steering request>"
}}"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": attribute_extraction_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        extraction_result = json.loads(response.choices[0].message.content)
        desired_attributes = extraction_result.get("attributes", [])
        attribute_reasoning = extraction_result.get("reasoning", "")

        all_features = []
        for attribute in desired_attributes:
            found_features = self._search_features_by_description(
                attribute, top_k=top_k_features
            )
            for feature_idx, relevance_score, description in found_features:
                all_features.append(
                    {
                        "attribute": attribute,
                        "feature_idx": feature_idx,
                        "relevance_score": relevance_score,
                        "feature_description": description,
                    }
                )

        return {
            "prompt": prompt,
            "steering_prompt": steering_prompt,
            "extracted_attributes": desired_attributes,
            "attribute_reasoning": attribute_reasoning,
            "candidate_features": all_features,
        }

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

    def get_random_features(self, count: int = 25) -> list[dict[str, Any]]:
        """Get random features with their descriptions for browsing."""
        import random

        if not self.explanations_by_index:
            return []

        valid_features = [
            (idx, desc)
            for idx, desc in self.explanations_by_index.items()
            if desc and isinstance(desc, str) and len(desc.strip()) > 10
        ]

        if not valid_features:
            return []

        sample_size = min(count, len(valid_features))
        sampled = random.sample(valid_features, sample_size)

        return [
            {
                "feature_idx": idx,
                "description": desc,
            }
            for idx, desc in sampled
        ]

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
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

        class AutoInterpretRequest(BaseModel):
            feature_idx: int
            example_texts: list[str]
            n_examples: int = Field(default=5, ge=1, le=20)

        class AutoSteerRequest(BaseModel):
            prompt: str
            desired_attributes: list[str]
            max_new_tokens: int = Field(default=100, ge=0, le=512)
            top_k_features: int = Field(default=3, ge=1, le=10)

        class AutoSteerFromPromptRequest(BaseModel):
            prompt: str
            steering_prompt: str
            max_new_tokens: int = Field(default=100, ge=0, le=512)
            top_k_features: int = Field(default=3, ge=1, le=10)

        class PreviewSteeringRequest(BaseModel):
            prompt: str
            steering_prompt: str
            top_k_features: int = Field(default=3, ge=1, le=10)

        api = FastAPI()

        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @api.get("/info")
        def info() -> dict[str, Any]:
            return self.get_model_info()

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

        @api.post("/auto_interpret")
        def auto_interpret(req: AutoInterpretRequest) -> dict[str, Any]:
            return self.auto_interpret_feature(
                req.feature_idx, req.example_texts, req.n_examples
            )

        @api.post("/auto_steer")
        def auto_steer_endpoint(req: AutoSteerRequest) -> dict[str, Any]:
            return self.auto_steer(
                req.prompt,
                req.desired_attributes,
                max_new_tokens=req.max_new_tokens,
                top_k_features=req.top_k_features,
            )

        @api.post("/auto_steer_from_prompt")
        def auto_steer_from_prompt_endpoint(
            req: AutoSteerFromPromptRequest,
        ) -> dict[str, Any]:
            return self.auto_steer_from_prompt(
                req.prompt,
                req.steering_prompt,
                max_new_tokens=req.max_new_tokens,
                top_k_features=req.top_k_features,
            )

        @api.post("/preview_steering")
        def preview_steering_endpoint(req: PreviewSteeringRequest) -> dict[str, Any]:
            return self.preview_steering(
                req.prompt,
                req.steering_prompt,
                top_k_features=req.top_k_features,
            )

        @api.get("/random_features")
        def random_features_endpoint(count: int = 25) -> dict[str, Any]:
            features = self.get_random_features(count=count)
            return {"features": features}

        return api
