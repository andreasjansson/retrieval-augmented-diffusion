import random
import re
import sys
import tempfile
import warnings
from typing import List

import numpy as np
import torch
from clip_retrieval.clip_back import load_index

from cog import BaseModel, BasePredictor, Input, Path
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder
from ldm.util import instantiate_from_config

sys.path.append("src/taming-transformers")
warnings.filterwarnings("ignore")

DATABASE = "monkey-island"
WIDTH = HEIGHT = 768
PREFIX = "A game scene from Monkey Island: "


def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 2 ** 32 - 1)
        print(f"Using random seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def knn_search(query: torch.Tensor, num_results: int, image_index):
    if query.dim() == 3:  # (b, 1, d)
        query = query.squeeze(1)  # need to expand to (b, d)
    query_embeddings = query.cpu().detach().numpy().astype(np.float32)
    # import ipdb; ipdb.set_trace()
    distances, indices, embeddings = image_index.search_and_reconstruct(
        query_embeddings, num_results
    )
    return distances, indices, embeddings


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    print(f"Loaded model from {ckpt}")
    return model


def build_searcher(database_name: str):
    image_index_path = Path(
        "data", "rdm", "faiss_indices", database_name, "image.index"
    )
    assert image_index_path.exists(), f"database at {image_index_path} does not exist"
    print(f"Loading semantic index from {image_index_path}")
    return {
        "image_index": load_index(
            str(image_index_path), enable_faiss_memory_mapping=True
        )
    }


class Predictor(BasePredictor):
    @torch.inference_mode()
    def setup(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.searchers = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        print(f"Loaded 1.4M param Retrieval Augmented Diffusion model to {self.device}")

        self.clip_text_model = FrozenCLIPTextEmbedder(device="cpu")
        print(f"Loaded Frozen CLIP Text Embedder to cpu")

        self.searcher = build_searcher(DATABASE)
        print(f"Loaded searcher for {DATABASE}")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            default="",
            description=f"Input prompt. Will be prefixed with '{PREFIX}'",
        ),
        scale: float = Input(
            default=5.0,
            description="Classifier-free unconditional scale for the sampler.",
        ),
        num_database_results: int = Input(
            default=10,
            description="The number of search results from the retrieval backend to guide the generation with.",
            ge=1,
            le=20,
        ),
        steps: int = Input(
            default=100,
            description="How many steps to run the model for. Using more will make generation take longer. 50 tends to work well.",
        ),
        seed: int = Input(
            default=-1,
            description="Seed for the random number generator. Set to -1 to use a random seed.",
        ),
    ) -> Path:
        if len(prompt.strip()) == 0:
            raise ValueError("No prompts provided")

        prompts = [PREFIX + prompt]

        set_seed(seed)
        print(f"Seed: {seed}")

        clip_text_embed = self.clip_text_model.encode(prompts)
        print(f"CLIP Text Embed: {clip_text_embed.shape}")

        sampler = PLMSSampler(self.model)

        _, _, result_embeddings = knn_search(
            query=clip_text_embed,
            num_results=num_database_results,
            image_index=self.searcher["image_index"],
        )
        result_embeddings = torch.from_numpy(result_embeddings).to(
            self.device
        )  # the input to the model is the result embeddings
        model_context = torch.cat(
            [clip_text_embed.to(self.device), result_embeddings.to(self.device)],
            dim=1,
        )

        unconditional_clip_embed = None
        if scale != 1.0:
            unconditional_clip_embed = torch.zeros_like(model_context)

        with self.model.ema_scope():
            shape = [
                16,
                HEIGHT // 16,
                WIDTH // 16,
            ]  # note: currently hardcoded for f16 model
            samples, _ = sampler.sample(
                S=steps,
                conditioning=model_context,
                batch_size=model_context.shape[0],
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=unconditional_clip_embed,
                eta=0.0,
            )
            decoded_generations = self.model.decode_first_stage(samples)
            decoded_generations = torch.clamp(
                (decoded_generations + 1.0) / 2.0, min=0.0, max=1.0
            )

            generation = 255.0 * rearrange(
                decoded_generations[0].cpu().numpy(), "c h w -> h w c"
            )
            img = (
                Image.fromarray(generation.astype(np.uint8))
                .convert("RGB")
                .resize([320, 320], Image.Resampling.NEAREST)
                .quantize(256, method=Image.Quantize.MAXCOVERAGE)
                .resize([640, 640], Image.Resampling.NEAREST)
            )
            out_path = "/tmp/out.png"
            img.save(out_path)
            return Path(out_path)
