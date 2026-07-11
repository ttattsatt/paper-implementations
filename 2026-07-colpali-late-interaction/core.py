"""
Core mechanism: ColPali late-interaction similarity maps on real
vidore_v3_industrial pages.

Loads ColPali, joins a corpus page to a real query via qrels, embeds both,
and saves per-token similarity maps (MaxSim visualized as a heatmap).
"""

from pathlib import Path
from typing import Optional, cast

import torch
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from datasets import load_dataset
from PIL import Image

MODEL_NAME = "vidore/colpali-v1.2"

corpus = load_dataset("vidore/vidore_v3_industrial", "corpus", split="test")
queries = load_dataset("vidore/vidore_v3_industrial", "queries", split="test")
qrels = load_dataset("vidore/vidore_v3_industrial", "qrels", split="test")


def load_model() -> tuple[ColPali, ColPaliProcessor]:
    """Load ColPali model + processor onto the best available device."""
    device = get_torch_device("auto")
    print(f"Using device: {device}")

    model = cast(
        ColPali,
        ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    ).eval()

    processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(MODEL_NAME))
    return model, processor


def query_for_corpus_id(corpus_id: int) -> Optional[str]:
    """Return one real query string whose qrels point at this corpus_id, or None."""
    match = qrels.filter(lambda x: x["corpus_id"] == corpus_id)
    if len(match) == 0:
        return None
    qid = match[0]["query_id"]
    q_match = queries.filter(lambda x: x["query_id"] == qid)
    return q_match[0]["query"] if len(q_match) > 0 else None


def generate_similarity_maps(
    model: ColPali,
    processor: ColPaliProcessor,
    image: Image.Image,
    query: str,
    out_dir: Path,
    tag: str,
) -> None:
    """Embed image + query, compute per-token similarity maps, save figures."""
    device = model.device

    batch_images = processor.process_images([image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)

    with torch.no_grad():
        image_embeddings = model.forward(**batch_images)
        query_embeddings = model.forward(**batch_queries)

    # model.patch_size (a property reading model.vision_tower.config.patch_size) can raise
    # AttributeError depending on how the model was loaded/dispatched — read the config directly.
    patch_size = model.config.vision_config.patch_size
    n_patches = processor.get_n_patches(image_size=image.size, patch_size=patch_size)
    image_mask = processor.get_image_mask(batch_images)

    batched_similarity_maps = get_similarity_maps_from_embeddings(
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        n_patches=n_patches,
        image_mask=image_mask,
    )
    similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

    query_content = processor.decode(batch_queries.input_ids[0])
    query_content = query_content.replace(processor.tokenizer.pad_token, "")
    query_content = query_content.replace(processor.query_augmentation_token, "").strip()
    query_tokens = processor.tokenizer.tokenize(query_content)

    out_dir.mkdir(parents=True, exist_ok=True)
    plots = plot_all_similarity_maps(
        image=image,
        query_tokens=query_tokens,
        similarity_maps=similarity_maps,
        figsize=(8, 8),
        show_colorbar=False,
        add_title=True,
    )
    for idx, (fig, _ax) in enumerate(plots):
        savepath = out_dir / f"baseline_{tag}_token{idx}.png"
        fig.savefig(savepath, bbox_inches="tight")
        print(f"Saved: {savepath}")


if __name__ == "__main__":
    # Corpus ids picked from previewing /tmp images — swap these for your chosen pages.
    chosen_corpus_ids = [23, 45, 67]

    model, processor = load_model()

    for corpus_id in chosen_corpus_ids:
        query = query_for_corpus_id(corpus_id)
        if query is None:
            print(f"Skipping corpus_id={corpus_id}: no qrels entry found")
            continue

        row = corpus[corpus_id]
        print(f"corpus_id={corpus_id} doc_id={row['doc_id']} query={query!r}")

        generate_similarity_maps(
            model=model,
            processor=processor,
            image=row["image"],
            query=query,
            out_dir=Path("figures"),
            tag=str(corpus_id),
        )
