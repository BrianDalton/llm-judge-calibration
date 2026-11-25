"""Local benchmark and source input helpers."""

import json

from models.benchmark_annotation import BenchmarkAnnotation


def read_benchmark(path):
    rows = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = BenchmarkAnnotation.model_validate_json(line)
            key = row.article_id, row.model_id
            if key in rows:
                raise ValueError(f"Duplicate benchmark key: {key}")
            if row.dimension != "consistency":
                raise ValueError(f"Expected consistency labels for {key}")
            rows[key] = row
    return rows


def read_sources(path, keys):
    sources = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["id"], row["model_id"]
            if key not in keys:
                continue
            if key in sources:
                raise ValueError(f"Duplicate source key: {key}")
            article, summary = row["text"], row["decoded"]
            publisher = row["filepath"].split("/")[1]
            if not all(isinstance(text, str) and text.strip() for text in (article, summary)):
                raise ValueError(f"Empty or invalid article or summary: {key}")
            if publisher not in {"cnn", "dailymail"}:
                raise ValueError(f"Unexpected publisher for {key}: {publisher}")
            sources[key] = article, summary, publisher
    missing = keys - sources.keys()
    if missing:
        raise ValueError(f"Missing paired sources for {len(missing)} articles, e.g. {min(missing)}")
    return sources
