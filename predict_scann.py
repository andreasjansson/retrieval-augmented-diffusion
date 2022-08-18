# import argparse
import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append("src/taming-transformers")

import glob
import os
import time
from itertools import islice
from multiprocessing import cpu_count
from functools import lru_cache

import clip
import numpy as np
import scann
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder
from ldm.util import instantiate_from_config, parallel_data_prefetch


import tempfile
from typing import List, Optional

# from clip_retrieval.clip_back import ParquetMetadataProvider, load_index, meta_to_dict
from cog import BasePredictor, Input, Path
from PIL import Image
from torch import nn


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


DATABASES = [
    # "openimages",
    "artbench-art_nouveau",
    "artbench-baroque",
    "artbench-expressionism",
    "artbench-impressionism",
    "artbench-post_impressionism",
    "artbench-realism",
    "artbench-romanticism",
    "artbench-renaissance",
    "artbench-surrealism",
    "artbench-ukiyo_e",
    "simulacra",
    "prompt_engineering",
]


class Searcher(object):
    def __init__(self, database, retriever_version="ViT-L/14"):
        assert database in DATABASES, f"{database} is not a valid database."
        self.database_name = database
        self.searcher_savedir = f"data/rdm/searchers/{self.database_name}"
        self.database_path = f"data/rdm/retrieval_databases/{self.database_name}"
        self.retriever = self.load_retriever(version=retriever_version)
        self.database = {"embedding": [], "img_id": [], "patch_coords": []}
        self.load_database()
        self.load_searcher()

    def train_searcher(self, k, metric="dot_product", searcher_savedir=None):

        print("Start training searcher")
        searcher = scann.scann_ops_pybind.builder(
            self.database["embedding"]
            / np.linalg.norm(self.database["embedding"], axis=1)[:, np.newaxis],
            k,
            metric,
        )
        self.searcher = searcher.score_brute_force().build()
        print("Finish training searcher")

        if searcher_savedir is not None:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_single_file(self, saved_embeddings):
        compressed = np.load(saved_embeddings)
        self.database = {key: compressed[key] for key in compressed.files}
        print("Finished loading of clip embeddings.")

    def load_multi_files(self, data_archive):
        out_data = {key: [] for key in self.database}
        for d in tqdm(
            data_archive,
            desc=f"Loading datapool from {len(data_archive)} individual files.",
        ):
            for key in d.files:
                out_data[key].append(d[key])

        return out_data

    def load_database(self):

        print(f'Load saved patch embedding from "{self.database_path}"')
        file_content = glob.glob(os.path.join(self.database_path, "*.npz"))

        if len(file_content) == 1:
            self.load_single_file(file_content[0])
        elif len(file_content) > 1:
            data = [np.load(f) for f in file_content]
            prefetched_data = parallel_data_prefetch(
                self.load_multi_files,
                data,
                n_proc=min(len(data), cpu_count()),
                target_data_type="dict",
            )

            self.database = {
                key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0]
                for key in self.database
            }
        else:
            raise ValueError(
                f'No npz-files in specified path "{self.database_path}" is this directory existing?'
            )

        print(
            f'Finished loading of retrieval database of length {self.database["embedding"].shape[0]}.'
        )

    def load_retriever(
        self,
        version="ViT-L/14",
    ):
        model = FrozenClipImageEmbedder(model=version)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def load_searcher(self):
        print(
            f"load searcher for database {self.database_name} from {self.searcher_savedir}"
        )
        self.searcher = scann.scann_ops_pybind.load_searcher(self.searcher_savedir)
        print("Finished loading searcher.")

    def search(self, x, k):
        if self.searcher is None and self.database["embedding"].shape[0] < 2e4:
            self.train_searcher(
                k
            )  # quickly fit searcher on the fly for small databases
        assert self.searcher is not None, "Cannot search with uninitialized searcher"
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(x.shape) == 3:
            x = x[:, 0]
        query_embeddings = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

        start = time.time()
        nns, distances = self.searcher.search_batched(
            query_embeddings, final_num_neighbors=k
        )
        end = time.time()

        out_embeddings = self.database["embedding"][nns]
        out_img_ids = self.database["img_id"][nns]
        out_pc = self.database["patch_coords"][nns]

        out = {
            "nn_embeddings": out_embeddings
            / np.linalg.norm(out_embeddings, axis=-1)[..., np.newaxis],
            "img_ids": out_img_ids,
            "patch_coords": out_pc,
            "queries": x,
            "exec_time": end - start,
            "nns": nns,
            "q_embeddings": query_embeddings,
        }

        return out

    def __call__(self, x, n):
        return self.search(x, n)


@lru_cache(maxsize=None)  # cache the model, so we don't have to load it every time
def load_clip(clip_model="ViT-L/14", use_jit=True, device="cpu"):
    clip_model, preprocess = clip.load(clip_model, device=device, jit=use_jit)
    return clip_model, preprocess


@torch.no_grad()
def encode_text_with_clip_model(
    text: str,
    clip_model: nn.Module,
    normalize: bool = True,
    device: str = "cpu",
):
    assert text is not None and len(text) > 0, "must provide text"
    tokens = clip.tokenize(text, truncate=True).to(device)
    clip_text_embed = clip_model.encode_text(tokens).to(device)
    if normalize:
        clip_text_embed /= clip_text_embed.norm(dim=-1, keepdim=True)
    if clip_text_embed.ndim == 2:
        clip_text_embed = clip_text_embed[:, None, :]
    return clip_text_embed


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
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


class Predictor(BasePredictor):
    def __init__(self):
        self.searchers = None

    @torch.inference_mode()
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        config = OmegaConf.load(f"configs/retrieval-augmented-diffusion/768x768.yaml")
        model = load_model_from_config(config, f"models/rdm/rdm768x768/model.ckpt")
        self.model = model.to(self.device)
        print(f"Loaded 1.4M param Retrieval Augmented Diffusion model to {self.device}")

        self.clip_text_encoder = FrozenCLIPTextEmbedder("ViT-L/14").to(self.device)
        use_jit = self.device.type.startswith("cuda")
        self.clip_model, _ = load_clip("ViT-L/14", use_jit=use_jit, device=self.device)
        print(f"Loaded clip model ViT-L/14 to {self.device} with use_jit={use_jit}")

        self.sampler = PLMSSampler(self.model)
        print("Using PLMS sampler")

        self.searchers = {}
        for db in DATABASES:
            print(f"Loading searcher for {db}")
            self.searchers[db] = Searcher(db)
            

    @torch.inference_mode()
    @torch.no_grad()
    def predict(
        self,
        prompt: str = Input(
            default="",
            description="model will try to generate this text.",
        ),
        database_name: str = Input(
            default="artbench-art_nouveau",
            description="Which database to use for the semantic search. Different databases have different capabilities.",
            choices=[  # TODO you have to copy this to the predict arg any time it is changed.
                # "openimages",
                "artbench-art_nouveau",
                "artbench-baroque",
                "artbench-expressionism",
                "artbench-impressionism",
                "artbench-post_impressionism",
                "artbench-realism",
                "artbench-romanticism",
                "artbench-renaissance",
                "artbench-surrealism",
                "artbench-ukiyo_e",
                "simulacra",
                "prompt_engineering",
            ],
        ),
        n_samples: int = Input(
            default=1,
            description="number of samples to generate",
        ),
        n_rows: int = Input(
            default=0,
            description="number of rows in the grid",
        ),
        scale: float = Input(
            default=5.0,
            description="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        ),
        use_neighbors: bool = Input(
            default=False,
            description="Whether to use neighbors in the semantic search",
        ),
        ddim_steps: int = Input(
            default=50,
            description="How many steps to run the model for. Using more will make generation take longer. 50 tends to work well.",
        ),
        height: int = Input(
            default=768,
            description="height of the image",
        ),
        width: int = Input(
            default=768,
            description="width of the image",
        ),
        knn: int = Input(
            default=10,
            description="number of nearest neighbors to use in the semantic search",
            ge=1,
            le=20,
        ),
        ddim_eta: float = Input(
            default=0.0,
            description="eta for the diffusion model",
            ge=0.0,
            le=1.0,
        ),
    ) -> List[Path]:
        outpath = tempfile.mkdtemp()
        batch_size = n_samples
        n_rows = n_rows if n_rows > 0 else batch_size
        assert prompt is not None
        data = [batch_size * [prompt]]
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        print(f"sampling scale for cfg is {scale:.2f}")

        searcher = self.searchers[database_name] if use_neighbors else None
        
        with self.model.ema_scope():
            # for n in trange(n_iter, desc="Sampling"):
            all_samples = list()
            for prompts in tqdm(data, desc="data"):
                print("sampling prompts:", prompts)
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = self.clip_text_encoder.encode(prompts)
                uc = None
                if searcher is not None:
                    nn_dict = searcher(c, knn)
                    c = torch.cat(
                        [c, torch.from_numpy(nn_dict["nn_embeddings"]).cuda()], dim=1
                    )
                if scale != 1.0:
                    uc = torch.zeros_like(c)
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                shape = [
                    16,
                    height // 16,
                    width // 16,
                ]  # note: currently hardcoded for f16 model
                samples_ddim, _ = self.sampler.sample(
                    S=ddim_steps,
                    conditioning=c,
                    batch_size=c.shape[0],
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                )

                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp(
                    (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                )

                for x_sample in x_samples_ddim:
                    x_sample = 255.0 * rearrange(
                        x_sample.cpu().numpy(), "c h w -> h w c"
                    )
                    Image.fromarray(x_sample.astype(np.uint8)).save(
                        os.path.join(sample_path, f"{base_count:05}.png")
                    )
                    base_count += 1
                all_samples.append(x_samples_ddim)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, "n b c h w -> (n b) c h w")
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(
                os.path.join(outpath, f"grid-{grid_count:04}.png")
            )
            grid_count += 1
