import random
import re
from cog import BasePredictor, Input, Path, BaseModel
import warnings

from clip_retrieval.clip_back import ParquetMetadataProvider, load_index, meta_to_dict
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

import sys
from functools import lru_cache

import tempfile
from typing import List

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

sys.path.append("src/taming-transformers")
warnings.filterwarnings("ignore")

DATABASE_NAMES = [
    "cars",
    "coco",
    "country211",
    "emotes",
    "faces",
    "food",
    "laion-aesthetic",
    "openimages",
    "pokemon",
    "prompt-engineer",
    "simulacra",
]
INIT_DATABASES = [
    "simulacra",
    "laion-aesthetic",
]  # start on cold boot, others will be loaded on first request
PROMPT_UPPER_BOUND = 8


def set_seed(seed):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class CaptionedImage(BaseModel):
    caption: str
    image: Path


def knn_search(query: torch.Tensor, num_results: int, image_index):
    query = query.to("cpu")  # TODO: remove this when we move to GPU
    if (
        query.ndim == 3
    ):  # query is in form (batch_size, 1, embedding_size), needs to be (batch_size, embedding_size)
        query = query.squeeze(1)
    query /= query.norm(dim=-1, keepdim=True)
    query_embeddings = query.squeeze(0).cpu().detach().numpy().astype(np.float32)
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


def map_to_metadata(
    indices, distances, num_images, metadata_provider, columns_to_return=["url"]
):
    results = []
    associated_metadata = metadata_provider.get(indices[:num_images])
    for key, (dist, ind) in enumerate(zip(distances, indices)):
        output = {}
        meta = None if key + 1 > len(associated_metadata) else associated_metadata[key]
        # convert_metadata_to_base64(meta) # TODO
        if meta is not None:
            output.update(meta_to_dict(meta))
        output["id"] = ind.item()
        output["similarity"] = dist.item()
        print(output)
        results.append(output)
    print(len(results))
    return results


def build_searcher(database_name: str):
    image_index_path = Path(
        "data", "rdm", "faiss_indices", database_name, "image.index"
    )
    assert image_index_path.exists(), f"database at {image_index_path} does not exist"
    print(f"Loading semantic index from {image_index_path}")
    metadata_path = Path("data", "rdm", "faiss_indices", database_name, "metadata")
    return {
        "image_index": load_index(
            str(image_index_path), enable_faiss_memory_mapping=True
        ),
        "metadata_provider": ParquetMetadataProvider(str(metadata_path))
        if metadata_path.exists()
        else None,
    }


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


class Predictor(BasePredictor):
    def __init__(self):
        self.searchers = None

    @torch.inference_mode()
    def setup(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        print(f"Loaded 1.4M param Retrieval Augmented Diffusion model to {self.device}")

        self.clip_text_model = FrozenCLIPTextEmbedder(device="cpu")
        print(f"Loaded Frozen CLIP Text Embedder to cpu")

        self.searchers = {
            database_name: build_searcher(database_name)
            for database_name in INIT_DATABASES
        }
        print(f"Loaded searchers for {INIT_DATABASES}")

        self.output_directory = Path(tempfile.mkdtemp())

    @torch.inference_mode()
    def predict(
        self,
        prompts: str = Input(
            default="",
            description="(batched) Text to generate. Use multiple prompts with a pipe `|` character. Limit 8 prompts/generations per run. Repeat prompts separated by `|` as needed.",
        ),
        database_name: str = Input(
            default="laion-aesthetic",
            description="Which database to use for the semantic search. Different databases have different capabilities.",
            choices=[  # TODO you have to copy this to the predict arg any time it is changed.
                "laion-aesthetic",
                "simulacra",
                "pokemon",
                "prompt-engineer",
                "emotes",
                "cars",
                "coco",
                "openimages",
                "country211",
                "faces",
                "food",
            ],
        ),
        scale: float = Input(
            default=5.0,
            description="Classifier-free unconditional scale for the sampler.",
        ),
        use_database: bool = Input(
            default=True,
            description="Disable to condition solely on the prompt without using retrieval-augmentation.",
        ),
        num_database_results: int = Input(
            default=10,
            description="The number of search results to guide the generation with. Using more will 'broaden' capabilities of the model at the risk of causing mode collapse or artifacting.",
            ge=1,
            le=20,
        ),
        height: int = Input(
            default=768, description="Desired height of generated images."
        ),
        width: int = Input(
            default=768, description="Desired width of generated images."
        ),
        steps: int = Input(
            default=50,
            description="How many steps to run the model for. Using more will make generation take longer. 50 tends to work well.",
        ),
        ddim_sampling: bool = Input(
            default=False,
            description="Use ddim sampling instead of the faster plms sampling.",
        ),
        ddim_eta: float = Input(
            default=0.0,
            description="The eta parameter for ddim sampling.",
        ),
        seed: int = Input(
            default=-1,
            description="Seed for the random number generator. Set to -1 to use a random seed.",
        ),
        full_batch: bool = Input(
            default=False,
            description="Use to always generate a full batch of 8 by repeating provided prompts.",
        ),
    ) -> List[CaptionedImage]:
        assert len(prompts) != 0, "Prompt must be longer than 0 characters"
        set_seed(seed)
        if ddim_sampling:
            print("Using ddim sampling")
            sampler = DDIMSampler(self.model)
        else:
            print("Using plms sampling")
            sampler = PLMSSampler(self.model)

        prompts = [prompt.strip() for prompt in prompts.split("|")]
        print(f"Found {len(prompts)} prompts: {prompts}")

        if len(prompts) < PROMPT_UPPER_BOUND and full_batch:
            prompts = [prompts[i % len(prompts)] for i in range(PROMPT_UPPER_BOUND)]

        clip_text_embed = self.clip_text_model.encode(prompts)

        if use_database:
            if database_name not in self.searchers:  # Load any new searchers
                self.searchers[database_name] = build_searcher(database_name)

            _, _, result_embeddings = knn_search(
                query=clip_text_embed,
                num_results=num_database_results,
                image_index=self.searchers[database_name]["image_index"],
            )
            result_embeddings = torch.from_numpy(result_embeddings).to(
                self.device
            )  # the input to the model is the result embeddings
            model_context = torch.cat(
                [clip_text_embed.to(self.device), result_embeddings.to(self.device)],
                dim=1,
            )
        else:
            print(f"warning: Using prompt only without any input from the database")
            model_context = clip_text_embed.to(self.device)

        unconditional_clip_embed = None
        if scale != 1.0:
            unconditional_clip_embed = torch.zeros_like(model_context)

        with self.model.ema_scope():
            shape = [
                16,
                height // 16,
                width // 16,
            ]  # note: currently hardcoded for f16 model
            samples, _ = sampler.sample(
                S=steps,
                conditioning=model_context,
                batch_size=model_context.shape[0],
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=unconditional_clip_embed,
                eta=ddim_eta,
            )
            decoded_generations = self.model.decode_first_stage(samples)
            decoded_generations = torch.clamp(
                (decoded_generations + 1.0) / 2.0, min=0.0, max=1.0
            )

            generation_paths = []
            captioned_generations = zip(decoded_generations, prompts)
            for idx, (generation, prompt) in enumerate(captioned_generations):
                generation = 255.0 * rearrange(
                    generation.cpu().numpy(), "c h w -> h w c"
                )
                generation_stub = f"sample_{idx:03d}__{slugify(prompt)}"
                x_sample_target_path = self.output_directory.joinpath(
                    f"{generation_stub}.png"
                )
                pil_image = Image.fromarray(generation.astype(np.uint8))
                pil_image.save(x_sample_target_path, "png")
                generation_paths.append(
                    CaptionedImage(caption=prompt, image=x_sample_target_path)
                )
            return generation_paths
