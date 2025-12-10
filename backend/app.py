import modal

import pandas as pd
import requests

MODEL_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_0/width_16k/canonical"

app = modal.App("smol-interp")

image = modal.Image.debian_slim().pip_install("transformer-lens", "sae-lens")

volume = modal.Volume.from_name("model-cache", create_if_missing=True)

# TODO: Try feature finding with ablation and attribution patching and DLA

@app.cls(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": volume},
    scaledown_window=300,
)
class Interp:
    @modal.enter()
    def load(self):
        import torch
        from transformer_lens import HookedTransformer
        from transformer_lens.hook_points import HookPoint
        from sae_lens import (SAE, ActivationsStore, HookedSAETransformer, LanguageModelSAERunnerConfig)

        self.model = HookedSAETransformer.from_pretrained_no_processing(
            MODEL_NAME,
            device="cuda",
            torch_dtype=torch.bfloat16,
            cache_dir="/cache",
        )

        self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=SAE_ID,
            device="cuda"
        )

    @modal.method()
    def run_inference(self, prompt: str):
        toks = self.model.to_tokens(prompt)
        logits = self.model(toks)

        return {
            "logits": logits.squeeze(0).tolist()
        }
    
    def get_auto_interp_neuronpedia(self, sae_release=SAE_RELEASE, sae_id=SAE_ID) -> pd.DataFrame:
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        release = get_pretrained_saes_directory()[sae_release]
        neuronpedia_id = release.neuronpedia_id[sae_id]

        url = "https://www.neuronpedia.org/api/explanation/export?modelId={}&saeId={}".format(
            *neuronpedia_id.split("/")
        )
        headers = {"Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
        return pd.DataFrame(response.json())
    
    def print_activated_features(self, prompt:str):
        import torch
        from sae_lens import SAE
        
        assert self.sae is not None and isinstance(self.sae, SAE), "SAE not loaded yet. Call load() first."
        
        logits, cache = self.model.run_with_cache_with_saes(prompt, saes=[self.sae])

        assert logits is not None and isinstance(logits, torch.Tensor), "Logits should be a tensor"
        
        top_logit_token_id = logits[0, -1].argmax(-1)
        top_logit_token_text = self.model.to_string(top_logit_token_id)
        print(f"Top predicted token from standard model logits: {top_logit_token_text!r}")

        cachename = f"{self.sae.cfg.metadata.hook_name}.hook_sae_acts_post"
        sae_acts_post = cache[cachename][0, -1, :]
        print([(act, idx) for act, idx in zip(*sae_acts_post.topk(5))])

    @modal.method()
    def get_model_info(self):
        return {
            "model_name": MODEL_NAME,
            "n_layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
        }


@app.local_entrypoint()
def main():
    model = Interp()

    info = model.get_model_info.remote()
    print(f"Model info: {info}")

if __name__ == "__main__":
    print("hi")
    model = Interp()
    print(model.get_auto_interp_neuronpedia().head)
