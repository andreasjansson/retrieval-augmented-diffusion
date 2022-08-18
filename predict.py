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
    image: Path
    caption: str


def knn_search(query: torch.Tensor, num_results: int, image_index):
    if query.dim() == 3:  # (b, 1, d)
        query = query.squeeze(1)  # need to expand to (b, d)
    query_embeddings = query.cpu().detach().numpy().astype(np.float32)
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
            description="(batched) Use up to 8 prompts by separating with a `|` character.",
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
            description="The number of search results from the retrieval backend to guide the generation with.",
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
    ) -> List[CaptionedImage]:
        set_seed(seed)
        if ddim_sampling:
            print("Using ddim sampling")
            sampler = DDIMSampler(self.model)
        else:
            print("Using plms sampling")
            sampler = PLMSSampler(self.model)

        prompts = prompts.split("|")
        if len(prompts) == 0:
            raise ValueError("No prompts provided")
        if len(prompts) > PROMPT_UPPER_BOUND:
            raise ValueError("You can only use up to 8 prompts. Try again using fewer `|` separators. Make sure to remove `|` characters from your captions if they are present.")
        print(f"Prompts: {prompts}")

        clip_text_embed = self.clip_text_model.encode(prompts)
        print(f"CLIP Text Embed: {clip_text_embed.shape}")

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
