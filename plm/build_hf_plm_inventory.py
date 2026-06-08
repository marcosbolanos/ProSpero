from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


SEARCH_QUERIES = [
    "protein",
    "proteins",
    "amino acid",
    "plm",
    "esm",
    "esm2",
    "esm3",
    "esmc",
    "protbert",
    "prot_bert",
    "prott5",
    "prostt5",
    "ankh",
    "progen",
    "protgpt2",
    "evo",
    "evodiff",
    "biological sequence",
]

KEYWORDS = [
    "protein",
    "proteins",
    "amino",
    "peptide",
    "esm",
    "esmc",
    "protbert",
    "prot_bert",
    "prott5",
    "prostt5",
    "ankh",
    "progen",
    "protgpt2",
    "evo",
    "evodiff",
    "sequence model",
]

EXCLUDE_SUBSTRINGS = [
    "gguf",
    "awq",
    "gptq",
    "int4",
    "int8",
    "4bit",
    "8bit",
    "lora",
    "qlora",
    "adapter",
    "instruction",
    "instruct",
    "chat",
    "reward",
    "rlhf",
    "classification",
    "regression",
    "localization",
    "token-classification",
    "sequence-classification",
    "ner",
    "qa",
    "question-answering",
    "text-classification",
    "sentiment",
    "folding",
]

FAMILY_PATTERNS: dict[str, list[str]] = {
    "esm3": [r"\besm3\b", r"esm3-"],
    "esmc": [r"\besmc\b", r"esmc-"],
    "esm2": [r"\besm2\b", r"esm2_", r"esm2-"],
    "esm1": [r"\besm1\b", r"esm-1", r"esm1b"],
    "protbert": [r"protbert", r"prot_bert"],
    "prott5": [r"prott5", r"prot_t5", r"prostt5"],
    "ankh": [r"\bankh\b"],
    "progen": [r"\bprogen\b", r"protgpt2"],
    "evo": [r"\bevo\b", r"evodiff", r"evolutionary"],
    "other": [],
}


@dataclass
class ModelEntry:
    model_id: str
    family: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    library_name: str | None
    private: bool
    gated: str | bool | None
    tags: list[str]
    last_modified: str | None
    search_hits: list[str]


def _clean_tags(tags: Any) -> list[str]:
    if not tags:
        return []
    return sorted({str(tag) for tag in tags})


def _contains_keyword(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in KEYWORDS)


def _classify_family(model_id: str, tags: list[str]) -> str:
    blob = f"{model_id} {' '.join(tags)}".lower()
    for family, patterns in FAMILY_PATTERNS.items():
        if family == "other":
            continue
        for pattern in patterns:
            if re.search(pattern, blob):
                return family
    return "other"


def _is_plm_candidate(model_id: str, tags: list[str], pipeline_tag: str | None) -> bool:
    blob = f"{model_id} {' '.join(tags)} {pipeline_tag or ''}".lower()
    if _contains_keyword(blob):
        return True
    if pipeline_tag in {"fill-mask", "feature-extraction", "text-generation", "masked-lm"}:
        if "protein" in blob or "esm" in blob or "prot" in blob:
            return True
    return False


def _is_runner_candidate(model_id: str, tags: list[str], pipeline_tag: str | None, family: str) -> bool:
    blob = f"{model_id} {' '.join(tags)} {pipeline_tag or ''}".lower()
    if any(term in blob for term in EXCLUDE_SUBSTRINGS):
        return False
    if family == "other":
        return False
    if family == "evo":
        if not ("evo-1" in blob or "evo-1.5" in blob or "evodiff" in blob):
            return False
    if pipeline_tag and pipeline_tag not in {
        "fill-mask",
        "feature-extraction",
        "text-generation",
        "masked-lm",
    }:
        return False
    return True


def _list_query_models(api: HfApi, query: str):
    return api.list_models(search=query, full=True)


def build_inventory() -> tuple[dict[str, Any], dict[str, Any]]:
    api = HfApi()
    by_id: dict[str, ModelEntry] = {}

    for query in SEARCH_QUERIES:
        for info in _list_query_models(api, query):
            model_id = str(info.id)
            tags = _clean_tags(getattr(info, "tags", []))
            pipeline_tag = getattr(info, "pipeline_tag", None)
            if not _is_plm_candidate(model_id, tags, pipeline_tag):
                continue
            entry = by_id.get(model_id)
            if entry is None:
                entry = ModelEntry(
                    model_id=model_id,
                    family=_classify_family(model_id, tags),
                    downloads=int(getattr(info, "downloads", 0) or 0),
                    likes=int(getattr(info, "likes", 0) or 0),
                    pipeline_tag=pipeline_tag,
                    library_name=getattr(info, "library_name", None),
                    private=bool(getattr(info, "private", False)),
                    gated=getattr(info, "gated", None),
                    tags=tags,
                    last_modified=(
                        info.last_modified.astimezone(timezone.utc).isoformat()
                        if getattr(info, "last_modified", None) is not None
                        else None
                    ),
                    search_hits=[query],
                )
                by_id[model_id] = entry
            else:
                if query not in entry.search_hits:
                    entry.search_hits.append(query)
                entry.downloads = max(entry.downloads, int(getattr(info, "downloads", 0) or 0))
                entry.likes = max(entry.likes, int(getattr(info, "likes", 0) or 0))
                entry.tags = sorted(set(entry.tags) | set(tags))
                if entry.pipeline_tag is None:
                    entry.pipeline_tag = pipeline_tag
                if entry.library_name is None:
                    entry.library_name = getattr(info, "library_name", None)
                if entry.last_modified is None and getattr(info, "last_modified", None) is not None:
                    entry.last_modified = info.last_modified.astimezone(timezone.utc).isoformat()

    raw_models = sorted((asdict(entry) for entry in by_id.values()), key=lambda x: x["model_id"])
    families: dict[str, int] = defaultdict(int)
    for row in raw_models:
        families[row["family"]] += 1

    raw_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "search_queries": SEARCH_QUERIES,
        "candidate_count": len(raw_models),
        "models": raw_models,
    }

    curated_models = [
        row
        for row in raw_models
        if row["family"] != "other" or ("protein" in " ".join(row["tags"]).lower())
    ]
    curated_models.sort(key=lambda row: (row["family"], -row["downloads"], row["model_id"]))

    runner_candidates = [
        row
        for row in curated_models
        if _is_runner_candidate(
            model_id=row["model_id"],
            tags=row.get("tags", []),
            pipeline_tag=row.get("pipeline_tag"),
            family=row.get("family", "other"),
        )
    ]
    runner_candidates.sort(
        key=lambda row: (row["family"], -int(row.get("downloads", 0)), row["model_id"])
    )

    curated_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Best-effort exhaustive HF PLM catalog built from multi-query search and heuristic protein filtering. "
            "Includes open and gated repos; training/fine-tuning status is not inferred."
        ),
        "candidate_count": len(raw_models),
        "curated_count": len(curated_models),
        "family_counts": dict(sorted(families.items())),
        "models": curated_models,
    }
    runner_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Subset of curated models intended for plug-and-play representation extraction "
            "with the universal Hugging Face runner."
        ),
        "count": len(runner_candidates),
        "models": runner_candidates,
    }
    return raw_payload, curated_payload, runner_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a data-driven Hugging Face PLM inventory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for inventory JSON outputs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_payload, curated_payload, runner_payload = build_inventory()

    raw_path = args.output_dir / "hf_protein_models_inventory_raw.json"
    curated_path = args.output_dir / "hf_protein_models_inventory_curated.json"
    runner_path = args.output_dir / "hf_protein_models_runner_candidates.json"

    raw_path.write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")
    curated_path.write_text(json.dumps(curated_payload, indent=2), encoding="utf-8")
    runner_path.write_text(json.dumps(runner_payload, indent=2), encoding="utf-8")

    print(f"Wrote raw inventory: {raw_path}")
    print(f"Wrote curated inventory: {curated_path}")
    print(f"Wrote runner candidates: {runner_path}")
    print(
        f"Candidates={raw_payload['candidate_count']} curated={curated_payload['curated_count']} "
        f"runner_candidates={runner_payload['count']} families={curated_payload['family_counts']}"
    )


if __name__ == "__main__":
    main()
