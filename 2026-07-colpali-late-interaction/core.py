"""
This is something I would like to import in scripts and (hopefull) notebooks
implementing the ColPali late-interaction similarity map visualizations on real
vidore_v3_industrial pages ( and more generally, any corpus with qrels and queries).

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


DIAGRAM_CONTENT_TYPES = {"Infographic", "Chart"}


def select_training_doc_ids(n_docs: int = 1) -> list[str]:
    """
    Rank doc_ids by how many qrels rows point at diagram-like content
    (Infographic/Chart, per qrels' content_type field) and return the top n_docs.

    This automates the manual "which doc looks diagram-heavy" step: a page's
    diagram-density is measured by how often real queries were annotated as
    needing an Infographic/Chart on that page, not just page-count.
    """
    corpus_id_to_doc_id = {row["corpus_id"]: row["doc_id"] for row in corpus}

    diagram_counts: dict[str, int] = {}
    for row in qrels:
        content_types = row["content_type"] or []
        if not DIAGRAM_CONTENT_TYPES.intersection(content_types):
            continue
        doc_id = corpus_id_to_doc_id.get(row["corpus_id"])
        if doc_id is None:
            continue
        diagram_counts[doc_id] = diagram_counts.get(doc_id, 0) + 1

    ranked = sorted(diagram_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _count in ranked[:n_docs]]


def build_train_slice(doc_ids: list[str], max_queries: int = 50, seed: int = 42) -> dict:
    """
    Build a small training slice: corpus pages restricted to doc_ids, and the
    queries whose qrels point at those pages (shuffled, then capped at max_queries).

    Full corpus/queries/qrels stay untouched for eval later — this only
    returns filtered views for training.
    """
    train_corpus = corpus.filter(lambda x: x["doc_id"] in doc_ids)
    train_corpus_ids = set(train_corpus["corpus_id"])

    train_qrel_ids = set(
        qrels.filter(lambda x: x["corpus_id"] in train_corpus_ids)["query_id"]
    )
    train_queries = queries.filter(lambda x: x["query_id"] in train_qrel_ids)
    train_queries = train_queries.shuffle(seed=seed)

    if len(train_queries) > max_queries:
        train_queries = train_queries.select(range(max_queries))

    print(f"Train slice: {len(train_corpus)} pages, {len(train_queries)} queries")
    return {"corpus": train_corpus, "queries": train_queries}


def preview_train_slice(doc_ids: list[str], out_dir: Path, n_preview: int = 8) -> None:
    """Save a handful of page images from the training doc(s) to out_dir for eyeballing."""
    train_corpus = corpus.filter(lambda x: x["doc_id"] in doc_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, len(train_corpus) // n_preview)
    for i in range(0, len(train_corpus), step):
        row = train_corpus[i]
        savepath = out_dir / f"preview_{row['doc_id']}_p{row['page_number_in_doc']}.png"
        row["image"].save(savepath)
        print(f"Saved: {savepath}")


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