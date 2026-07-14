"""Canonical data preparation and deterministic split manifests.

The training script deliberately keeps model concerns out of this module.  This
module owns the facts that must remain stable across training, calibration, and
later artifact-promotion sessions: normalized records, grouping keys, global
deduplication, deterministic partition assignment, and split-manifest
validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


CANONICAL_RECORD_SCHEMA_VERSION = "canonical_record/v1"
CANONICAL_DATA_MANIFEST_SCHEMA_VERSION = "canonical_data_manifest/v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "split_manifest/v1"
NORMALIZATION_VERSION = "text_nfkc_whitespace_casefold/v1"
GROUPING_VERSION = "canonical_url_article_event/v1"
PARTITIONS = ("train", "development", "calibration", "locked_test")
CLASSIFIER_INPUT_CONTRACT = "target_sentence_marked/v1"
CANONICAL_OPINION_LABELS = (
    "Entirely factual",
    "Somewhat factual but also opinionated",
    "Expresses writer's opinion",
)
CANONICAL_OPINION_LABEL2ID = {
    label: index for index, label in enumerate(CANONICAL_OPINION_LABELS)
}
OPINION_STYLE_LABELS = ("objective_style", "opinionated_style")
OPINION_STYLE_LABEL2ID = {
    label: index for index, label in enumerate(OPINION_STYLE_LABELS)
}
SOURCE_OPINION_TIER_TO_STYLE = {
    CANONICAL_OPINION_LABELS[0]: OPINION_STYLE_LABELS[0],
    CANONICAL_OPINION_LABELS[1]: OPINION_STYLE_LABELS[1],
    CANONICAL_OPINION_LABELS[2]: OPINION_STYLE_LABELS[1],
}
_OPINION_APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\uff07": "'",
    }
)
_OPINION_LABEL_ALIASES = {
    label.casefold(): label for label in CANONICAL_OPINION_LABELS
}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "_ga",
    "_gl",
}


class ManifestValidationError(ValueError):
    """Raised when a split or canonical-data manifest is not trustworthy."""


def normalize_text(value: Any) -> str:
    """Return a display-preserving, whitespace-normalized text value."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split()).strip()


def serialize_classifier_input(value: Any) -> str:
    """Serialize the production classifier input identically in every stage."""

    text = normalize_text(value)
    if not text:
        raise ValueError("Classifier target text must not be empty.")
    return f"[TARGET]\n{text}"


def normalized_hash_text(value: Any) -> str:
    """Return the canonical text form used for equality and hashes."""

    return normalize_text(value).casefold()


def sha256_text(value: Any) -> str:
    return hashlib.sha256(normalized_hash_text(value).encode("utf-8")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_url(value: Any) -> str | None:
    """Canonicalize a valid article URL without dropping substantive identity.

    Only HTTP(S) URLs with a hostname are accepted.  URL fragments, default
    ports, and known tracking parameters are removed; all other query fields
    remain and are sorted to make parameter order irrelevant.
    """

    raw = normalize_text(value)
    if not raw:
        return None
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.casefold().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        return None
    netloc = host
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = parsed.path or "/"
    query_pairs = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.casefold()
        if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
            continue
        query_pairs.append((key, item))
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_bias_label(value: Any) -> str:
    label = normalize_text(value)
    aliases = {
        "0": "Non-biased",
        "0.0": "Non-biased",
        "nonbiased": "Non-biased",
        "non-biased": "Non-biased",
        "1": "Biased",
        "1.0": "Biased",
        "biased": "Biased",
    }
    return aliases.get(label.casefold(), label)


def normalize_opinion_label(value: Any) -> str:
    """Normalize known BABE opinion tiers while preserving unknown labels.

    ``normalize_text`` applies NFKC, whitespace collapse, and trimming. This
    extra translation makes common curly apostrophe forms equivalent to the
    straight apostrophe in the canonical writer-opinion label before a
    case-insensitive alias lookup.
    """

    label = normalize_text(value).translate(_OPINION_APOSTROPHE_TRANSLATION)
    return _OPINION_LABEL_ALIASES.get(label.casefold(), label)


def bias_label_value(label: str) -> int | None:
    if label == "Biased":
        return 1
    if label == "Non-biased":
        return 0
    return None


def opinion_label_value(label: str) -> int | None:
    """Return the legacy three-tier source-label identifier."""

    return CANONICAL_OPINION_LABEL2ID.get(label)


def opinion_style_label_name(label: str) -> str | None:
    """Map a normalized BABE source tier to the binary writing-style target."""

    return SOURCE_OPINION_TIER_TO_STYLE.get(label)


def opinion_style_label_value(label: str) -> int | None:
    style_name = opinion_style_label_name(label)
    if style_name is None:
        return None
    return OPINION_STYLE_LABEL2ID[style_name]


def _first_existing_column(
    columns: Iterable[str], requested: str | None, aliases: Sequence[str] = ()
) -> str | None:
    available = set(columns)
    if requested and requested in available:
        return requested
    for alias in aliases:
        if alias in available:
            return alias
    return None


def _row_value(row: pd.Series, column: str | None) -> str:
    if not column:
        return ""
    return normalize_text(row.get(column, ""))


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def source_dataset_for_file(path: str | Path | None) -> str:
    name = Path(path).name.casefold() if path else ""
    if "mbic" in name:
        return "MBIC"
    if "basil" in name:
        return "BASIL"
    return "BABE"


def canonicalize_records(
    frame: pd.DataFrame,
    *,
    source_dataset: str = "BABE",
    text_column: str = "text",
    bias_label_column: str | None = "label_bias",
    opinion_label_column: str | None = "label_opinion",
    group_column: str | None = "news_link",
    article_id_column: str | None = None,
    event_column: str | None = None,
    source_record_id_column: str | None = None,
    source_file_column: str = "source_file",
    input_split_column: str = "dataset_split",
) -> pd.DataFrame:
    """Convert a source-shaped dataframe into canonical sentence records.

    The function is intentionally permissive: missing task labels are retained
    as null so a future masked-label trainer can consume them.  The current
    flat-head trainer requests both labels during its cleaning pass.
    """

    text_column = _first_existing_column(frame.columns, text_column, ("sentence", "text"))
    if not text_column:
        raise KeyError("Could not resolve a text column for canonicalization.")
    bias_column = _first_existing_column(frame.columns, bias_label_column, ("label_bias", "label"))
    opinion_column = _first_existing_column(frame.columns, opinion_label_column, ("label_opinion",))
    url_column = _first_existing_column(frame.columns, article_id_column or group_column, ("news_link", "url", "link"))
    event_column = _first_existing_column(
        frame.columns,
        event_column,
        ("event_id", "story_id", "event", "story"),
    )
    source_record_id_column = _first_existing_column(
        frame.columns,
        source_record_id_column,
        ("uuid", "id", "record_id", "sentence_id"),
    )
    article_column = _first_existing_column(
        frame.columns,
        None,
        ("article", "article_text", "full_text", "article_body", "story_text"),
    )
    outlet_column = _first_existing_column(frame.columns, None, ("outlet", "source", "publisher"))
    topic_column = _first_existing_column(frame.columns, None, ("topic", "category"))
    report_relevant_column = _first_existing_column(
        frame.columns, None, ("report_relevant", "is_report_relevant")
    )

    records: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        text = _row_value(row, text_column)
        source_file = _row_value(row, source_file_column) or "<in_memory>"
        source_record_id = _row_value(row, source_record_id_column) or f"{source_file}:{row_index}"
        row_source_dataset = _row_value(row, "source_dataset") or source_dataset
        row_source_dataset = row_source_dataset.upper()
        canonical_url = canonicalize_url(_row_value(row, url_column))
        event_id = _row_value(row, event_column).casefold() or None
        article_text = _row_value(row, article_column)
        article_hash = sha256_text(article_text) if article_text else None
        text_hash = sha256_text(text)
        if row_source_dataset == "BASIL" and event_id:
            canonical_group_id = f"basil:event:{event_id.casefold()}"
            grouping_method = "basil_event"
        elif canonical_url:
            canonical_group_id = f"url:{canonical_url}"
            grouping_method = "canonical_url"
        elif article_hash:
            canonical_group_id = f"article:{article_hash}"
            grouping_method = "article_hash"
        else:
            canonical_group_id = f"text:{text_hash}"
            grouping_method = "text_hash_fallback"

        bias_label_name = normalize_bias_label(_row_value(row, bias_column))
        opinion_label_name = normalize_opinion_label(_row_value(row, opinion_column))
        style_label_name = opinion_style_label_name(opinion_label_name)
        no_agreement = bias_label_name.casefold().startswith("no agreement") or opinion_label_name.casefold().startswith("no agreement")
        if no_agreement:
            annotation_state = "no_agreement"
        elif bias_label_name or opinion_label_name:
            annotation_state = "single_annotated"
        else:
            annotation_state = "unlabeled"
        report_raw = _row_value(row, report_relevant_column).casefold()
        report_relevant: bool | None
        if report_raw in {"1", "true", "yes"}:
            report_relevant = True
        elif report_raw in {"0", "false", "no"}:
            report_relevant = False
        else:
            report_relevant = None
        identity = {
            "source_dataset": row_source_dataset,
            "source_record_id": source_record_id,
            "source_file": source_file,
            "text_hash": text_hash,
        }
        record_id = f"rec:{canonical_json_sha256(identity)}"
        records.append(
            {
                "record_id": record_id,
                "source_dataset": row_source_dataset,
                "source_record_id": source_record_id,
                "source_file": source_file,
                "input_split": _row_value(row, input_split_column) or "train",
                "canonical_group_id": canonical_group_id,
                "grouping_method": grouping_method,
                "event_id": event_id,
                "canonical_url": canonical_url,
                "article_hash": article_hash,
                "outlet": _row_value(row, outlet_column) or None,
                "topic": _row_value(row, topic_column) or None,
                "text": text,
                "text_hash": text_hash,
                "bias_label_name": bias_label_name or None,
                "opinion_label_name": opinion_label_name or None,
                "bias_label": bias_label_value(bias_label_name),
                "opinion_label": opinion_label_value(opinion_label_name),
                "source_opinion_label": opinion_label_name or None,
                "opinion_style_label_name": style_label_name,
                "opinion_style_label": (
                    OPINION_STYLE_LABEL2ID[style_label_name]
                    if style_label_name is not None
                    else None
                ),
                "report_relevant": report_relevant,
                "annotation_state": annotation_state,
                "annotation_provenance": json.dumps(
                    [{"source_dataset": row_source_dataset, "source_record_id": source_record_id}],
                    sort_keys=True,
                ),
                "license": None,
                "source_record_ids": json.dumps([source_record_id]),
                "source_datasets": json.dumps([row_source_dataset]),
            }
        )
    return pd.DataFrame.from_records(records)


def merge_mbic_annotations(
    canonical_records: pd.DataFrame,
    mbic_records: pd.DataFrame,
    cleaning_stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Merge MBIC provenance into matching canonical rows without new rows."""

    if mbic_records.empty:
        return canonical_records
    stats = cleaning_stats if cleaning_stats is not None else {}
    stats.setdefault("mbic_matched", 0)
    stats.setdefault("mbic_unmatched", 0)
    result = canonical_records.copy()
    by_text: dict[str, list[int]] = defaultdict(list)
    for index, row in result.iterrows():
        by_text[str(row["text_hash"])].append(index)
    for _, mbic in mbic_records.iterrows():
        candidates = by_text.get(str(mbic["text_hash"]), [])
        mbic_url = mbic.get("canonical_url")
        if mbic_url:
            candidates = [
                index
                for index in candidates
                if result.at[index, "canonical_url"] == mbic_url
            ]
        if len(candidates) != 1:
            stats["mbic_unmatched"] += 1
            continue
        index = candidates[0]
        provenance = json.loads(str(result.at[index, "annotation_provenance"]))
        provenance.append(
            {
                "source_dataset": "MBIC",
                "source_record_id": str(mbic["source_record_id"]),
            }
        )
        source_ids = json.loads(str(result.at[index, "source_record_ids"]))
        source_ids.append(str(mbic["source_record_id"]))
        source_datasets = sorted(set(json.loads(str(result.at[index, "source_datasets"])) + ["MBIC"]))
        result.at[index, "annotation_provenance"] = json.dumps(provenance, sort_keys=True)
        result.at[index, "source_record_ids"] = json.dumps(sorted(set(source_ids)))
        result.at[index, "source_datasets"] = json.dumps(source_datasets)
        stats["mbic_matched"] += 1
    return result


def _increment(stats: dict[str, Any], key: str, count: int = 1) -> None:
    stats["excluded"][key] = int(stats["excluded"].get(key, 0)) + count


def clean_and_dedupe_records(
    records: pd.DataFrame,
    *,
    include_no_agreement: bool = False,
    require_both_labels: bool = True,
    global_dedupe: bool = True,
    opinion_target_column: str = "opinion_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply global exclusion/deduplication with reproducible representatives."""

    stats: dict[str, Any] = {
        "input_rows": int(len(records)),
        "excluded": {},
        "output_rows": 0,
        "grouping_method_counts": {},
        "source_counts": {},
    }
    if records.empty:
        return records.copy(), stats
    if opinion_target_column not in records.columns:
        raise KeyError(
            f"Opinion target column {opinion_target_column!r} is not present."
        )
    result = records.copy()
    empty_text = result["text"].fillna("").eq("")
    _increment(stats, "empty_text", int(empty_text.sum()))
    result = result[~empty_text].copy()
    if not include_no_agreement:
        no_agreement = result["annotation_state"].eq("no_agreement")
        _increment(stats, "no_agreement", int(no_agreement.sum()))
        result = result[~no_agreement].copy()
    if require_both_labels:
        missing_labels = (
            result["bias_label"].isna()
            | result[opinion_target_column].isna()
        )
        _increment(stats, "missing_required_label", int(missing_labels.sum()))
        result = result[~missing_labels].copy()

    conflict_hashes: set[str] = set()
    if global_dedupe:
        for text_hash, group in result.groupby("text_hash", sort=True):
            for label_column in ("bias_label", opinion_target_column):
                values = {value for value in group[label_column].tolist() if pd.notna(value)}
                if len(values) > 1:
                    conflict_hashes.add(str(text_hash))
                    break
    if conflict_hashes:
        conflict_mask = result["text_hash"].isin(conflict_hashes)
        _increment(stats, "conflicting_text_labels", int(conflict_mask.sum()))
        result = result[~conflict_mask].copy()

    if not result.empty and global_dedupe:
        result = result.sort_values(
            ["text_hash", "source_dataset", "source_file", "source_record_id", "record_id"],
            kind="mergesort",
        )
        representatives: list[pd.Series] = []
        for _, group in result.groupby("text_hash", sort=True):
            representative = group.iloc[0].copy()
            source_ids: list[str] = []
            datasets: list[str] = []
            provenance: list[dict[str, Any]] = []
            for _, candidate in group.iterrows():
                source_ids.extend(json.loads(str(candidate["source_record_ids"])))
                datasets.extend(json.loads(str(candidate["source_datasets"])))
                provenance.extend(json.loads(str(candidate["annotation_provenance"])))
            representative["source_record_ids"] = json.dumps(sorted(set(source_ids)))
            representative["source_datasets"] = json.dumps(sorted(set(datasets)))
            representative["annotation_provenance"] = json.dumps(
                sorted(provenance, key=lambda value: (value.get("source_dataset", ""), value.get("source_record_id", ""))),
                sort_keys=True,
            )
            representatives.append(representative)
            _increment(stats, "global_duplicate", len(group) - 1)
        result = pd.DataFrame(representatives)

    result = result.reset_index(drop=True)
    stats["output_rows"] = int(len(result))
    stats["grouping_method_counts"] = {
        str(key): int(value)
        for key, value in result["grouping_method"].value_counts(dropna=False).sort_index().items()
    }
    stats["source_counts"] = {
        str(key): int(value)
        for key, value in result["source_dataset"].value_counts(dropna=False).sort_index().items()
    }
    return result, stats


def build_canonical_data_manifest(
    records: pd.DataFrame,
    cleaning_stats: Mapping[str, Any],
    *,
    source_files: Sequence[Path] = (),
) -> dict[str, Any]:
    file_entries = []
    for path in sorted({Path(item).resolve() for item in source_files}, key=lambda item: str(item)):
        entry: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
        file_entries.append(entry)
    record_entries = []
    for _, row in records.sort_values("record_id", kind="mergesort").iterrows():
        record_entries.append(
            {
                "record_id": str(row["record_id"]),
                "source_dataset": str(row["source_dataset"]),
                "canonical_group_id": str(row["canonical_group_id"]),
                "event_id": row["event_id"] if pd.notna(row["event_id"]) else None,
                "canonical_url": row["canonical_url"] if pd.notna(row["canonical_url"]) else None,
                "text_hash": str(row["text_hash"]),
                "grouping_method": str(row["grouping_method"]),
            }
        )
    manifest = {
        "schema_version": CANONICAL_DATA_MANIFEST_SCHEMA_VERSION,
        "canonical_record_schema_version": CANONICAL_RECORD_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "grouping_version": GROUPING_VERSION,
        "source_files": file_entries,
        "cleaning_stats": dict(cleaning_stats),
        "record_count": len(record_entries),
        "records": record_entries,
    }
    manifest["canonical_data_manifest_sha256"] = canonical_json_sha256(manifest)
    validate_canonical_data_manifest(manifest)
    return manifest


def validate_canonical_data_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_canonical_data_manifest_sha256: str | None = None,
) -> None:
    """Validate the immutable identity and minimum structure of a data manifest."""

    if manifest.get("schema_version") != CANONICAL_DATA_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError("Unsupported canonical data manifest schema version.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ManifestValidationError("Canonical data manifest has no records.")
    required = {
        "record_id",
        "source_dataset",
        "canonical_group_id",
        "text_hash",
        "grouping_method",
    }
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ManifestValidationError("Canonical data manifest contains an invalid record.")
        missing = required.difference(record)
        if missing:
            raise ManifestValidationError(
                "Canonical data manifest records are missing fields: "
                f"{sorted(missing)}"
            )
        record_id = str(record["record_id"])
        if record_id in record_ids:
            raise ManifestValidationError("Canonical data manifest contains duplicate record IDs.")
        record_ids.add(record_id)
    if int(manifest.get("record_count", -1)) != len(records):
        raise ManifestValidationError("Canonical data manifest record_count does not match records.")
    identity_payload = copy.deepcopy(dict(manifest))
    observed_hash = identity_payload.pop("canonical_data_manifest_sha256", None)
    expected_hash = canonical_json_sha256(identity_payload)
    if observed_hash != expected_hash:
        raise ManifestValidationError(
            "canonical_data_manifest_sha256 does not match manifest content."
        )
    if (
        expected_canonical_data_manifest_sha256 is not None
        and observed_hash != expected_canonical_data_manifest_sha256
    ):
        raise ManifestValidationError(
            "Canonical data manifest does not bind the expected canonical data hash."
        )


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _connected_split_units(records: pd.DataFrame) -> list[dict[str, Any]]:
    union_find = _UnionFind(len(records))
    seen: dict[tuple[str, str], int] = {}
    for position, (_, row) in enumerate(records.reset_index(drop=True).iterrows()):
        values = {
            "canonical_group_id": row.get("canonical_group_id"),
            "event_id": row.get("event_id"),
            "text_hash": row.get("text_hash"),
        }
        for kind, value in values.items():
            if value is None or _is_missing(value) or not str(value):
                continue
            key = (kind, str(value))
            previous = seen.get(key)
            if previous is None:
                seen[key] = position
            else:
                union_find.union(position, previous)
    members: dict[int, list[int]] = defaultdict(list)
    for position in range(len(records)):
        members[union_find.find(position)].append(position)
    normalized = records.reset_index(drop=True)
    units: list[dict[str, Any]] = []
    for positions in members.values():
        rows = normalized.iloc[positions]
        keys = sorted(
            f"{kind}:{value}"
            for kind in ("canonical_group_id", "event_id", "text_hash")
            for value in rows[kind].dropna().astype(str).unique().tolist()
            if value
        )
        split_unit_id = f"unit:{canonical_json_sha256(keys)}"
        units.append(
            {
                "split_unit_id": split_unit_id,
                "positions": positions,
                "row_count": len(positions),
                "bias_counts": Counter(int(value) for value in rows["bias_label"].dropna().tolist()),
                "opinion_counts": Counter(int(value) for value in rows["opinion_label"].dropna().tolist()),
            }
        )
    return units


def _hash_tiebreak(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    result = {partition: float(fractions.get(partition, 0.0)) for partition in PARTITIONS}
    if any(value < 0.0 or value >= 1.0 for value in result.values()):
        raise ValueError("Split fractions must be in [0, 1).")
    if result["train"] == 0.0:
        result["train"] = 1.0 - sum(
            result[partition] for partition in PARTITIONS if partition != "train"
        )
    if result["train"] <= 0.0 or not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must leave a positive train fraction and sum to one.")
    return result


def _normalized_target_error(observed: float, target: float) -> float:
    return abs(observed - target) / max(1.0, target)


def _allocate_units(
    units: Sequence[dict[str, Any]],
    fractions: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    active = [partition for partition in PARTITIONS if fractions[partition] > 0.0]
    if len(units) < len(active):
        raise ValueError(
            "Not enough independent article/event/text groups to populate every requested partition."
        )
    total_rows = sum(int(unit["row_count"]) for unit in units)
    bias_total: Counter[int] = Counter()
    opinion_total: Counter[int] = Counter()
    for unit in units:
        bias_total.update(unit["bias_counts"])
        opinion_total.update(unit["opinion_counts"])
    target_rows = {partition: total_rows * fractions[partition] for partition in PARTITIONS}
    target_bias = {
        partition: {label: count * fractions[partition] for label, count in bias_total.items()}
        for partition in PARTITIONS
    }
    target_opinion = {
        partition: {label: count * fractions[partition] for label, count in opinion_total.items()}
        for partition in PARTITIONS
    }
    current_rows = Counter()
    current_bias: dict[str, Counter[int]] = {partition: Counter() for partition in PARTITIONS}
    current_opinion: dict[str, Counter[int]] = {partition: Counter() for partition in PARTITIONS}
    assignments: dict[str, str] = {}
    ordered_units = sorted(
        units,
        key=lambda unit: (-int(unit["row_count"]), _hash_tiebreak(seed, str(unit["split_unit_id"]))),
    )
    for unit_index, unit in enumerate(ordered_units):
        empty_partitions = [
            partition for partition in active if current_rows[partition] == 0
        ]
        remaining_units = len(ordered_units) - unit_index
        eligible_partitions = active
        if empty_partitions and remaining_units == len(empty_partitions):
            # Reserve the final available units for partitions that would
            # otherwise be empty, without distorting the balance objective.
            eligible_partitions = empty_partitions
        scores: list[tuple[float, str]] = []
        for partition in eligible_partitions:
            new_rows = current_rows[partition] + int(unit["row_count"])
            score = _normalized_target_error(
                new_rows, target_rows[partition]
            ) - _normalized_target_error(
                current_rows[partition], target_rows[partition]
            )
            for label, count in unit["bias_counts"].items():
                expected = target_bias[partition][label]
                score += 0.35 * (
                    _normalized_target_error(
                        current_bias[partition][label] + count, expected
                    )
                    - _normalized_target_error(
                        current_bias[partition][label], expected
                    )
                )
            for label, count in unit["opinion_counts"].items():
                expected = target_opinion[partition][label]
                score += 0.35 * (
                    _normalized_target_error(
                        current_opinion[partition][label] + count, expected
                    )
                    - _normalized_target_error(
                        current_opinion[partition][label], expected
                    )
                )
            scores.append((score, partition))
        _, selected = min(scores, key=lambda item: (item[0], PARTITIONS.index(item[1])))
        assignments[str(unit["split_unit_id"])] = selected
        current_rows[selected] += int(unit["row_count"])
        current_bias[selected].update(unit["bias_counts"])
        current_opinion[selected].update(unit["opinion_counts"])
    return assignments


def _overlap_values(left: pd.DataFrame, right: pd.DataFrame, column: str) -> list[str]:
    left_values = {str(value) for value in left[column].dropna().tolist() if str(value)}
    right_values = {str(value) for value in right[column].dropna().tolist() if str(value)}
    return sorted(left_values.intersection(right_values))


def build_leakage_report(records: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"pairs": {}, "has_overlap": False}
    for left_index, left_name in enumerate(PARTITIONS):
        left = records[records["partition"] == left_name]
        for right_name in PARTITIONS[left_index + 1 :]:
            right = records[records["partition"] == right_name]
            pair: dict[str, Any] = {}
            for column in ("canonical_group_id", "event_id", "text_hash"):
                values = _overlap_values(left, right, column)
                pair[column] = {"count": len(values), "sample": values[:10]}
                report["has_overlap"] = report["has_overlap"] or bool(values)
            report["pairs"][f"{left_name}__{right_name}"] = pair
    return report


def _partition_summary(records: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for partition in PARTITIONS:
        subset = records[records["partition"] == partition]
        summary[partition] = {
            "record_count": int(len(subset)),
            "group_count": int(subset["canonical_group_id"].nunique(dropna=True)),
            "split_unit_count": int(subset["split_unit_id"].nunique(dropna=True)),
            "source_counts": {
                str(key): int(value)
                for key, value in subset["source_dataset"].value_counts(dropna=False).sort_index().items()
            },
            "outlet_counts": {
                str(key): int(value)
                for key, value in subset["outlet"].fillna("<missing>").value_counts().sort_index().items()
            },
            "topic_counts": {
                str(key): int(value)
                for key, value in subset["topic"].fillna("<missing>").value_counts().sort_index().items()
            },
            "bias_label_support": {
                str(int(key)): int(value)
                for key, value in subset["bias_label"].dropna().astype(int).value_counts().sort_index().items()
            },
            "opinion_label_support": {
                str(int(key)): int(value)
                for key, value in subset["opinion_label"].dropna().astype(int).value_counts().sort_index().items()
            },
        }
    return summary


def _locked_test_support(records: pd.DataFrame) -> dict[str, Any]:
    test = records[records["partition"] == "locked_test"]
    bias_positive = int((test["bias_label"] == 1).sum())
    middle_opinion = int((test["opinion_label"] == 1).sum())
    minimums = {"bias_positive": 500, "middle_opinion": 300}
    return {
        "observed": {"bias_positive": bias_positive, "middle_opinion": middle_opinion},
        "minimums": minimums,
        "passed": bias_positive >= minimums["bias_positive"] and middle_opinion >= minimums["middle_opinion"],
    }


def _manifest_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("created_at_utc", None)
    payload.pop("split_content_sha256", None)
    return payload


def build_group_split_manifest(
    records: pd.DataFrame,
    *,
    seed: int,
    validation_size: float = 0.15,
    calibration_size: float = 0.15,
    test_size: float = 0.20,
    canonical_data_manifest_sha256: str,
    cleaning_stats: Mapping[str, Any],
    leakage_policy: str = "error",
    split_strategy: str = "group",
    created_at_utc: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign group-disjoint four-way partitions and produce split_manifest/v1."""

    if leakage_policy not in {"error", "warn"}:
        raise ValueError("leakage_policy must be 'error' or 'warn'.")
    if records.empty:
        raise ValueError("Cannot split an empty canonical dataset.")
    fractions = _validate_fractions(
        {
            "train": 0.0,
            "development": validation_size,
            "calibration": calibration_size,
            "locked_test": test_size,
        }
    )
    units = _connected_split_units(records)
    assignments = _allocate_units(units, fractions, seed=seed)
    split_unit_by_position: dict[int, str] = {}
    for unit in units:
        for position in unit["positions"]:
            split_unit_by_position[position] = str(unit["split_unit_id"])
    assigned = records.reset_index(drop=True).copy()
    assigned["split_unit_id"] = [split_unit_by_position[index] for index in range(len(assigned))]
    assigned["partition"] = assigned["split_unit_id"].map(assignments)
    leakage_report = build_leakage_report(assigned)
    if leakage_policy == "error" and leakage_report["has_overlap"]:
        raise ManifestValidationError("Group-aware split contains prohibited leakage overlap.")
    summary = _partition_summary(assigned)
    manifest_assignments = []
    for _, row in assigned.sort_values("record_id", kind="mergesort").iterrows():
        manifest_assignments.append(
            {
                "record_id": str(row["record_id"]),
                "partition": str(row["partition"]),
                "split_unit_id": str(row["split_unit_id"]),
                "canonical_group_id": str(row["canonical_group_id"]),
                "event_id": row["event_id"] if pd.notna(row["event_id"]) else None,
                "text_hash": str(row["text_hash"]),
                "source_dataset": str(row["source_dataset"]),
                "grouping_method": str(row["grouping_method"]),
            }
        )
    group_assignments = [
        {"canonical_group_id": group, "partition": partition}
        for group, partition in sorted(
            {
                (str(row["canonical_group_id"]), str(row["partition"]))
                for _, row in assigned.iterrows()
            }
        )
    ]
    actual_fractions = {
        partition: (float(summary[partition]["record_count"]) / len(assigned))
        for partition in PARTITIONS
    }
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "canonical_record_schema_version": CANONICAL_RECORD_SCHEMA_VERSION,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": int(seed),
        "split_strategy": split_strategy,
        "normalization_version": NORMALIZATION_VERSION,
        "grouping_version": GROUPING_VERSION,
        "canonical_data_manifest_sha256": canonical_data_manifest_sha256,
        "leakage_policy": leakage_policy,
        "requested_fractions": fractions,
        "actual_fractions": actual_fractions,
        "cleaning_stats": dict(cleaning_stats),
        "assignments": manifest_assignments,
        "canonical_group_assignments": group_assignments,
        "partition_summaries": summary,
        "leakage_report": leakage_report,
        "locked_test_support": _locked_test_support(assigned),
    }
    manifest["split_content_sha256"] = canonical_json_sha256(_manifest_identity_payload(manifest))
    validate_split_manifest(manifest, expected_canonical_data_manifest_sha256=canonical_data_manifest_sha256)
    return assigned, manifest


def build_legacy_split_manifest(
    records: pd.DataFrame,
    partitions_by_record_id: Mapping[str, str],
    *,
    seed: int,
    canonical_data_manifest_sha256: str,
    cleaning_stats: Mapping[str, Any],
    leakage_policy: str = "warn",
    created_at_utc: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Serialize an explicit row-level compatibility split.

    Legacy manifests intentionally do not claim article/event/text disjointness.
    Their leakage report remains complete and a warning policy marks them as
    unsuitable for promotion.
    """

    if leakage_policy != "warn":
        raise ValueError("Legacy split manifests require leakage_policy='warn'.")
    assigned = records.copy().reset_index(drop=True)
    assigned["partition"] = assigned["record_id"].map(partitions_by_record_id)
    if assigned["partition"].isna().any():
        raise ManifestValidationError("Legacy split did not assign every canonical record.")
    if not set(assigned["partition"]).issubset(set(PARTITIONS)):
        raise ManifestValidationError("Legacy split contains an unknown partition.")
    assigned["split_unit_id"] = assigned["record_id"].map(lambda value: f"legacy:{value}")
    summary = _partition_summary(assigned)
    manifest_assignments = []
    for _, row in assigned.sort_values("record_id", kind="mergesort").iterrows():
        manifest_assignments.append(
            {
                "record_id": str(row["record_id"]),
                "partition": str(row["partition"]),
                "split_unit_id": str(row["split_unit_id"]),
                "canonical_group_id": str(row["canonical_group_id"]),
                "event_id": row["event_id"] if pd.notna(row["event_id"]) else None,
                "text_hash": str(row["text_hash"]),
                "source_dataset": str(row["source_dataset"]),
                "grouping_method": str(row["grouping_method"]),
            }
        )
    actual_fractions = {
        partition: (float(summary[partition]["record_count"]) / len(assigned))
        for partition in PARTITIONS
    }
    manifest: dict[str, Any] = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "canonical_record_schema_version": CANONICAL_RECORD_SCHEMA_VERSION,
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": int(seed),
        "split_strategy": "legacy",
        "normalization_version": NORMALIZATION_VERSION,
        "grouping_version": GROUPING_VERSION,
        "canonical_data_manifest_sha256": canonical_data_manifest_sha256,
        "leakage_policy": leakage_policy,
        "requested_fractions": None,
        "actual_fractions": actual_fractions,
        "cleaning_stats": dict(cleaning_stats),
        "assignments": manifest_assignments,
        "canonical_group_assignments": [],
        "partition_summaries": summary,
        "leakage_report": build_leakage_report(assigned),
        "locked_test_support": _locked_test_support(assigned),
    }
    manifest["split_content_sha256"] = canonical_json_sha256(_manifest_identity_payload(manifest))
    validate_split_manifest(manifest, expected_canonical_data_manifest_sha256=canonical_data_manifest_sha256)
    return assigned, manifest


def _manifest_records(manifest: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_records(list(manifest.get("assignments", [])))


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_canonical_data_manifest_sha256: str | None = None,
) -> None:
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError("Unsupported split manifest schema version.")
    if expected_canonical_data_manifest_sha256 and manifest.get("canonical_data_manifest_sha256") != expected_canonical_data_manifest_sha256:
        raise ManifestValidationError("Split manifest does not bind the expected canonical data manifest.")
    assignments = _manifest_records(manifest)
    if assignments.empty:
        raise ManifestValidationError("Split manifest has no assignments.")
    required = {"record_id", "partition", "split_unit_id", "canonical_group_id", "text_hash"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ManifestValidationError(f"Split manifest assignments are missing fields: {sorted(missing)}")
    if assignments["record_id"].duplicated().any():
        raise ManifestValidationError("A record is assigned more than once.")
    if not set(assignments["partition"]).issubset(set(PARTITIONS)):
        raise ManifestValidationError("Split manifest contains an unknown partition.")
    strict_disjointness = manifest.get("leakage_policy") == "error"
    if strict_disjointness and (assignments.groupby("canonical_group_id")["partition"].nunique() > 1).any():
        raise ManifestValidationError("A canonical article group spans multiple partitions.")
    if strict_disjointness:
        for column in ("event_id", "text_hash"):
            if column not in assignments.columns:
                continue
            nonempty = assignments[assignments[column].notna() & assignments[column].astype(str).ne("")]
            if (nonempty.groupby(column)["partition"].nunique() > 1).any():
                raise ManifestValidationError(f"A {column} value spans multiple partitions.")
    reconstructed = assignments.copy()
    report = build_leakage_report(reconstructed)
    if manifest.get("leakage_policy") == "error" and report["has_overlap"]:
        raise ManifestValidationError("Split manifest violates leakage_policy=error.")
    calibration = set(assignments.loc[assignments["partition"] == "calibration", "record_id"])
    locked_test = set(assignments.loc[assignments["partition"] == "locked_test", "record_id"])
    if calibration.intersection(locked_test):
        raise ManifestValidationError("Calibration and locked-test records overlap.")
    expected_hash = manifest.get("split_content_sha256")
    if expected_hash and expected_hash != canonical_json_sha256(_manifest_identity_payload(manifest)):
        raise ManifestValidationError("split_content_sha256 does not match manifest content.")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_split_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def split_partition_assignment_sha256(
    manifest: Mapping[str, Any], partition: str
) -> str:
    """Return a stable fingerprint for one validated split partition."""

    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    validate_split_manifest(manifest)
    fields = (
        "record_id",
        "partition",
        "split_unit_id",
        "canonical_group_id",
        "event_id",
        "text_hash",
        "source_dataset",
        "grouping_method",
    )
    assignments: list[dict[str, Any]] = []
    for entry in sorted(
        (
            item
            for item in manifest["assignments"]
            if str(item["partition"]) == partition
        ),
        key=lambda item: str(item["record_id"]),
    ):
        assignments.append({field: entry.get(field) for field in fields})
    return canonical_json_sha256(
        {
            "partition": partition,
            "assignments": assignments,
        }
    )


def records_for_partition(
    records: pd.DataFrame,
    manifest: Mapping[str, Any],
    partition: str,
) -> pd.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError(f"Unknown partition: {partition}")
    validate_split_manifest(manifest)
    assignment = {
        str(entry["record_id"]): str(entry["partition"])
        for entry in manifest["assignments"]
    }
    selected = records[records["record_id"].map(assignment).eq(partition)].copy()
    expected = {record_id for record_id, assigned_partition in assignment.items() if assigned_partition == partition}
    actual = set(selected["record_id"].tolist())
    if expected != actual:
        raise ManifestValidationError(
            f"Canonical records do not match the split manifest for {partition}."
        )
    return selected.reset_index(drop=True)
