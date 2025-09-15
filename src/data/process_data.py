import json, os, re, random
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Tuple

RAW_PATH = Path("data/raw/ori_pqal.json")
OUT_DIR = Path("data/processed")
SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train/val/test
VALID_DECISIONS = {"yes", "no", "maybe"}


def load_raw(fp: Path) -> Dict[str, Any]:
    with fp.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    # collapse weird whitespace while preserving sentence boundaries
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def join_contexts(ctxs: List[str]) -> str:
    ctxs = [normalize(c) for c in (ctxs or []) if c and isinstance(c, str)]
    return "\n\n".join([c for c in ctxs if c])


def generative_record(pid: str, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": pid,
        "question": normalize(item.get("QUESTION", "")),
        "context": join_contexts(item.get("CONTEXTS", [])),
        "answer": normalize(item.get("LONG_ANSWER", "")),
        "year": item.get("YEAR", None),
        "labels": item.get("LABELS", []),
        "final_decision": normalize(item.get("final_decision", "")).lower(),
    }


def classification_record(pid: str, item: Dict[str, Any]) -> Dict[str, Any]:
    # binary/ternary (yes/no/maybe) target using final_decision
    fd = normalize(item.get("final_decision", "")).lower()
    return {
        "id": pid,
        "question": normalize(item.get("QUESTION", "")),
        "context": join_contexts(item.get("CONTEXTS", [])),
        "target": fd if fd in VALID_DECISIONS else None,
        "year": item.get("YEAR", None),
        "labels": item.get("LABELS", []),
    }


def drop_incorrect_data(
    records: List[Dict[str, Any]], mode: str
) -> List[Dict[str, Any]]:
    cleaned = []
    for r in records:
        # require question + context minimally
        if not r.get("question") or not r.get("context"):
            continue
        if mode == "gen":
            if not r.get("answer"):
                continue
        elif mode == "clf":
            if r.get("target") not in VALID_DECISIONS:
                continue
        cleaned.append(r)
    return cleaned


def dedupe_by_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in records:
        rid = r.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out


def stratified_split(
    records: List[Dict[str, Any]], key: str
) -> Tuple[List, List, List]:
    """Simple label-stratified split: keeps label proportions by 'key'."""
    random.seed(SEED)
    buckets = {}
    for r in records:
        label = r.get(key, "unknown")
        buckets.setdefault(label, []).append(r)

    train, val, test = [], [], []
    for label, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(SPLIT_RATIOS[0] * n)
        n_val = int(SPLIT_RATIOS[1] * n)
        t_slice = items[:n_train]
        v_slice = items[n_train : n_train + n_val]
        s_slice = items[n_train + n_val :]
        train.extend(t_slice)
        val.extend(v_slice)
        test.extend(s_slice)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_json_processed(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    raw = load_raw(RAW_PATH)

    # Build records
    gen_records = []
    clf_records = []
    for pid, item in raw.items():
        if not isinstance(item, dict):
            continue
        gen_records.append(generative_record(pid, item))
        clf_records.append(classification_record(pid, item))

    # Clean + dedupe
    gen_records = dedupe_by_id(drop_incorrect_data(gen_records, mode="gen"))
    clf_records = dedupe_by_id(drop_incorrect_data(clf_records, mode="clf"))

    # Splits (stratify on different keys)
    gen_train, gen_val, gen_test = stratified_split(gen_records, key="final_decision")
    clf_train, clf_val, clf_test = stratified_split(clf_records, key="target")

    # Save
    save_json_processed(OUT_DIR / "generative" / "train.jsonl", gen_train)
    save_json_processed(OUT_DIR / "generative" / "val.jsonl", gen_val)
    save_json_processed(OUT_DIR / "generative" / "test.jsonl", gen_test)

    save_json_processed(OUT_DIR / "classification" / "train.jsonl", clf_train)
    save_json_processed(OUT_DIR / "classification" / "val.jsonl", clf_val)
    save_json_processed(OUT_DIR / "classification" / "test.jsonl", clf_test)

    print("\nWrote processed files to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
