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
from colpali_engine.data.dataset import ColPaliEngineDataset

import math

import torch
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from datasets import load_dataset
from PIL import Image
from colpali_engine.data.dataset import ColPaliEngineDataset


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


def load_colpali_train_model() -> ColPali:
    """
    Wrapper matching the `(): core.func` config resolution convention, used
    in place of AllPurposeWrapper for the `model:` block in train_config.yaml.

    On a real T4 run, `model.lm_head.weight` was the parameter left on the
    meta device after from_pretrained() (ColPali's _keys_to_ignore_on_load_missing
    explicitly excludes it from checkpoint loading, since the LM head isn't
    used for retrieval — so nothing ever streams a value into it).
    transformers constructs the whole model under a meta-device context,
    then streams in real checkpoint weights per-parameter as it finds
    matching keys; any parameter with no matching key (or explicitly
    ignored, like lm_head.weight here) stays meta permanently.

    Confirmed empirically (transformers 5.13.1) that low_cpu_mem_usage=False
    alone does NOT prevent this — the meta tensor is still present
    immediately after from_pretrained() returns, before any .to(device)
    call. This contradicts older reports (e.g. GitHub issue #29423) where
    that alone was sufficient, suggesting the meta-device construction
    became unconditional in more recent transformers versions.

    Fix: after loading, snapshot every correctly-loaded (non-meta) parameter
    to CPU RAM, call to_empty() to materialize the whole model on the target
    device (this wipes ALL parameters to fresh uninitialized memory, not
    just the meta ones — confirmed directly), restore the snapshotted real
    weights (popping each from the snapshot dict as it's restored, to avoid
    holding two full copies of a 3B-param model in memory at once — an
    earlier version of this function snapshotted on GPU and OOM-killed on a
    T4), then re-run weight init scoped only to the specific submodule(s)
    that were actually meta, leaving every pretrained weight untouched.
    Verified structurally correct on a CPU dummy model reproducing this
    exact base-model-plus-ignored-head pattern (own testing, not fully
    GPU-verified against the real 3B checkpoint due to no local GPU access —
    the meta-tensor detection and materialization logic matched the real
    run's reported "Materializing 1 meta tensor(s): ['model.lm_head.weight']"
    exactly, but the subsequent OOM fix here is untested beyond this).
    """
    device = get_torch_device("auto")
    model = ColPali.from_pretrained(
        "vidore/colpaligemma-3b-pt-448-base",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    meta_param_names = [n for n, p in model.named_parameters() if p.is_meta]
    if not meta_param_names:
        return model.to(device)

    print(f"Materializing {len(meta_param_names)} meta tensor(s): {meta_param_names}")

    # model.to(device) raises NotImplementedError the instant it hits ANY
    # meta parameter, regardless of how many others are fine — confirmed
    # directly, this isn't a "some tensors ok, some not" partial failure.
    # to_empty() is the only way to move a model containing meta tensors,
    # but it wipes every parameter (not just the meta ones) to fresh
    # uninitialized memory. Snapshotting all non-meta tensors on GPU before
    # to_empty() briefly holds ~2x the model in GPU memory and OOM-killed
    # on a T4 — snapshot on CPU RAM instead, and free each tensor from the
    # snapshot the moment it's restored, keeping peak GPU memory to ~1x.
    real_tensors_cpu = {
        n: p.detach().to("cpu").clone()
        for n, p in model.named_parameters()
        if n not in meta_param_names
    }

    model = model.to_empty(device=device)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in real_tensors_cpu:
                param.copy_(real_tensors_cpu.pop(name).to(device))

    # Only the modules that owned a meta parameter still hold uninitialized
    # memory at this point — re-init just those, not the whole model, so
    # pretrained weights elsewhere are left untouched.
    touched_modules = {name.rpartition(".")[0] for name in meta_param_names}
    for module_path in touched_modules:
        module = model.get_submodule(module_path) if module_path else model
        module.apply(model._init_weights)

    
    # Fix: embed_scale is a buffer (not a parameter), so it isn't
    # captured by the named_parameters()-based snapshot/restore and
    # gets left at its to_empty()-initialized value of 0.0, which
    # zeroes out every embedding and propagates NaN through the model.
    embed_layer = model.model.model.language_model.embed_tokens
    embed_layer.embed_scale = torch.tensor(
        math.sqrt(model.config.text_config.hidden_size),
        device=model.device,
        dtype=model.dtype,
    )

    return model


DIAGRAM_CONTENT_TYPES = {"Infographic", "Chart"}


def select_training_doc_ids(n_docs: int = 1) -> list[str]:
    """
    Rank doc_ids by how many qrels rows point at diagram-like content
    (Infographic/Chart, per qrels' content_type field) and return the top n_docs.

    This automates the manual "which doc looks diagram-heavy" step: a page's
    diagram-density is measured by how often real queries were annotated as
    needing an Infographic/Chart on that page, not just page-count.
    """
    # select_columns first: iterating full `corpus` rows decodes the image
    # column per-row and hangs/stalls badly on Colab (confirmed: .filter()
    # and row iteration over the image column can take minutes to never-finish
    # vs <1s once the image column is dropped from the view).
    corpus_ids_and_docs = corpus.select_columns(["corpus_id", "doc_id"])
    corpus_id_to_doc_id = {row["corpus_id"]: row["doc_id"] for row in corpus_ids_and_docs}

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


def build_train_slice(doc_ids: list[str], max_queries: int = 50, n_eval: int = 5, seed: int = 42) -> dict:
    """
    Build a small training/eval slice: corpus pages restricted to doc_ids,
    English-only queries whose qrels point at those pages (shuffled, capped
    at max_queries), each query tagged with its single positive corpus_id
    (highest qrels score if multiple relevant pages exist), then split into
    train/eval.

    Full corpus/queries/qrels stay untouched for Phase 5 eval — this only
    returns filtered, flattened views for training.
    """
    # Same image-decode-hang issue as select_training_doc_ids(): filter on an
    # image-free projection to find matching row indices, then .select() those
    # indices from the original `corpus` so the image column survives in the
    # output without ever being decoded during the filter predicate itself.
    corpus_ids_and_docs = corpus.select_columns(["corpus_id", "doc_id"])
    match_indices = [
        i for i, row in enumerate(corpus_ids_and_docs) if row["doc_id"] in doc_ids
    ]
    train_corpus = corpus.select(match_indices)
    train_corpus_ids = set(train_corpus["corpus_id"])

    relevant_qrels = qrels.filter(lambda x: x["corpus_id"] in train_corpus_ids)

    best_positive: dict[str, tuple[int, int]] = {}
    for row in relevant_qrels:
        qid, cid, score = row["query_id"], row["corpus_id"], row["score"]
        if qid not in best_positive or score > best_positive[qid][0]:
            best_positive[qid] = (score, cid)

    train_qrel_ids = set(best_positive.keys())
    train_queries = queries.filter(
        lambda x: x["query_id"] in train_qrel_ids and x["language"] == "english"
    )
    train_queries = train_queries.shuffle(seed=seed)

    if len(train_queries) > max_queries:
        train_queries = train_queries.select(range(max_queries))

    train_queries = train_queries.map(
        lambda x: {"positive_corpus_id": best_positive[x["query_id"]][1]}
    )

    n_eval = min(n_eval, len(train_queries) - 1) if len(train_queries) > n_eval else 0
    if n_eval > 0:
        eval_queries = train_queries.select(range(len(train_queries) - n_eval, len(train_queries)))
        train_queries = train_queries.select(range(len(train_queries) - n_eval))
    else:
        eval_queries = None

    print(f"Train slice: {len(train_corpus)} pages, {len(train_queries)} train queries"
          + (f", {len(eval_queries)} eval queries" if eval_queries else ", no eval split"))

    return {"corpus": train_corpus, "queries": train_queries, "eval_queries": eval_queries}

def preview_train_slice(doc_ids: list[str], out_dir: Path, n_preview: int = 8) -> None:
    """Save a handful of page images from the training doc(s) to out_dir for eyeballing."""
    corpus_ids_and_docs = corpus.select_columns(["corpus_id", "doc_id"])
    match_indices = [
        i for i, row in enumerate(corpus_ids_and_docs) if row["doc_id"] in doc_ids
    ]
    train_corpus = corpus.select(match_indices)
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


def build_colpali_train_dataset(train_slice: dict) -> ColPaliEngineDataset:
    """
    Flatten {corpus, queries} into one row-per-query dataset with the
    query's single ground-truth positive page image attached directly,
    matching ColPaliEngineDataset(dataset, pos_target_column_name="image").
    """
    corpus_by_id = {row["corpus_id"]: row["image"] for row in train_slice["corpus"]}

    def attach_image(example):
        return {"image": corpus_by_id[example["positive_corpus_id"]]}

    flat = train_slice["queries"].map(attach_image)
    return ColPaliEngineDataset(flat, pos_target_column_name="image")

def load_colpali_train_slice() -> ColPaliEngineDataset:
    """Wrapper matching load_train_set()'s signature for YAML !ext / (): use."""
    doc_ids = select_training_doc_ids(n_docs=1)
    s = build_train_slice(doc_ids, max_queries=50, n_eval=5)
    return build_colpali_train_dataset({"corpus": s["corpus"], "queries": s["queries"]})


def load_colpali_eval_slice() -> ColPaliEngineDataset:
    """Same idea for eval_dataset."""
    doc_ids = select_training_doc_ids(n_docs=1)
    s = build_train_slice(doc_ids, max_queries=50, n_eval=5)
    return build_colpali_train_dataset({"corpus": s["corpus"], "queries": s["eval_queries"]})