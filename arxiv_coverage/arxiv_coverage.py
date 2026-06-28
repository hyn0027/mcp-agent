#!/usr/bin/env python3
"""
Estimate arXiv coverage of 2025 *accepted* papers from:
  - NeurIPS, ICML, ICLR     (via OpenReview API v2)
  - ACL, EMNLP, NAACL       (via ACL Anthology XML; Main + Findings)

Sampling: for each venue we pick min(100, n) papers uniformly at random
with a fixed seed; the sample is cached so resuming uses the same papers.

For each sampled paper we query the arXiv API and accept a match when the
normalized title similarity is >= 0.9.

The script is resumable: per-venue partial results are persisted to
cache/<venue>.arxiv.json after every completed request, and re-running
skips already-processed papers. Progress is also written to progress.log.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import signal
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
from rapidfuzz import fuzz
from tqdm import tqdm

HERE = Path(__file__).parent
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)
LOG_PATH = HERE / "progress.log"

UA = "arxiv-coverage-script/1.0 (mailto:anonymous@example.com)"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA

# --- logging ----------------------------------------------------------------

logger = logging.getLogger("arxiv_coverage")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_PATH)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_sh = logging.StreamHandler(sys.stderr)
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
logger.addHandler(_sh)

# Global stop flag for graceful shutdown on SIGINT/SIGTERM.
STOP = threading.Event()


def _install_signal_handlers():
    def handler(signum, frame):
        if not STOP.is_set():
            logger.warning("Received signal %d; finishing in-flight work and "
                           "shutting down. Press Ctrl+C again to force.", signum)
            STOP.set()
        else:
            logger.error("Forced exit.")
            os._exit(130)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, handler)
        except ValueError:
            pass  # not in main thread


# --- atomic write -----------------------------------------------------------

def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


# -----------------------------------------------------------------------------
# Title normalization + matching
# -----------------------------------------------------------------------------

def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def title_similarity(a: str, b: str) -> float:
    return fuzz.ratio(normalize_title(a), normalize_title(b)) / 100.0


# -----------------------------------------------------------------------------
# OpenReview (NeurIPS / ICML / ICLR) - API v2
# -----------------------------------------------------------------------------

OPENREVIEW_API = "https://api2.openreview.net/notes"

OPENREVIEW_VENUES = {
    "NeurIPS-2025": "NeurIPS.cc/2025/Conference",
    "ICML-2025":    "ICML.cc/2025/Conference",
    "ICLR-2025":    "ICLR.cc/2025/Conference",
}


def fetch_openreview(venue_id: str) -> list[dict]:
    """Accepted papers only: filter by content.venueid (only set on acceptance)."""
    papers: list[dict] = []
    offset = 0
    limit = 1000
    while True:
        params = {
            "content.venueid": venue_id,
            "limit": limit,
            "offset": offset,
        }
        r = SESSION.get(OPENREVIEW_API, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json().get("notes", [])
        if not batch:
            break
        for n in batch:
            content = n.get("content", {}) or {}
            title = content.get("title")
            if isinstance(title, dict):
                title = title.get("value")
            venue = content.get("venue")
            if isinstance(venue, dict):
                venue = venue.get("value")
            if not title:
                continue
            papers.append({"id": n.get("id"), "title": title, "venue": venue})
        if len(batch) < limit:
            break
        offset += limit
    return papers


# -----------------------------------------------------------------------------
# ACL Anthology (ACL / EMNLP / NAACL) - XML from GitHub
# -----------------------------------------------------------------------------

ANTHOLOGY_XML = ("https://raw.githubusercontent.com/acl-org/acl-anthology/"
                 "master/data/xml/{event}.xml")

# Mapping: our venue label -> (anthology event file, set of acceptable volume ids)
ANTHOLOGY_VENUES = {
    "ACL-2025":   ("2025.acl",   {"long", "short", "findings"}),
    "EMNLP-2025": ("2025.emnlp", {"main", "long", "short", "findings"}),
    "NAACL-2025": ("2025.naacl", {"long", "short", "findings"}),
}


def fetch_anthology(event: str, volumes: set[str]) -> list[dict]:
    url = ANTHOLOGY_XML.format(event=event)
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    papers: list[dict] = []
    collection_id = root.get("id", event)
    for vol in root.findall("volume"):
        vol_id = vol.get("id", "")
        if vol_id not in volumes:
            continue
        for paper in vol.findall("paper"):
            if paper.get("id") == "0":  # frontmatter
                continue
            title_el = paper.find("title")
            if title_el is None:
                continue
            title = "".join(title_el.itertext()).strip()
            if not title:
                continue
            paper_full_id = f"{collection_id}-{vol_id}.{paper.get('id')}"
            papers.append({"id": paper_full_id, "title": title, "venue": vol_id})
    return papers


# -----------------------------------------------------------------------------
# arXiv search
# -----------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}


class ArxivQueryError(Exception):
    pass


def arxiv_search_title(title: str, retries: int = 6) -> list[tuple[str, str]]:
    """Returns candidate (id, title) list. Raises ArxivQueryError if the
    request never succeeded (so caller can avoid caching a false negative)."""
    clean = re.sub(r"[^A-Za-z0-9 ]+", " ", title).strip()
    if not clean:
        return []
    words = clean.split()[:25]
    query = " ".join(words)
    params = {
        "search_query": f'ti:"{query}"',
        "start": 0,
        "max_results": 5,
    }
    last_err: Exception | None = None
    logger.debug("arxiv GET %s?search_query=%s", ARXIV_API, params["search_query"])
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = SESSION.get(ARXIV_API, params=params, timeout=30)
            dt = time.time() - t0
            body_len = len(r.content) if r.content else 0
            logger.debug("  attempt=%d status=%s bytes=%d elapsed=%.2fs url=%s",
                         attempt, r.status_code, body_len, dt, r.url)
            if r.status_code == 200 and r.content:
                # Log first chunk of XML for visibility
                snippet = r.text[:300].replace("\n", " ")
                logger.debug("  body[:300]=%s", snippet)
                try:
                    root = ET.fromstring(r.content)
                except ET.ParseError as e:
                    last_err = e
                    backoff = min(60, 2 ** attempt)
                    logger.warning("  XML parse error: %s; sleeping %.1fs", e, backoff)
                    time.sleep(backoff)
                    continue
                out: list[tuple[str, str]] = []
                for entry in root.findall("a:entry", ATOM_NS):
                    id_el = entry.find("a:id", ATOM_NS)
                    title_el = entry.find("a:title", ATOM_NS)
                    if id_el is None or title_el is None:
                        continue
                    arxiv_id = id_el.text.strip().rsplit("/", 1)[-1]
                    out.append((arxiv_id, title_el.text.strip()))
                logger.debug("  parsed %d candidates", len(out))
                return out
            last_err = RuntimeError(f"HTTP {r.status_code}")
            if r.status_code in (429, 503):
                retry_after = r.headers.get("Retry-After")
                backoff = min(180, 10 * (2 ** attempt))
                if retry_after and retry_after.isdigit():
                    backoff = max(backoff, int(retry_after))
                logger.warning("  RATE-LIMITED status=%d Retry-After=%s; "
                               "sleeping %.1fs (attempt %d/%d)",
                               r.status_code, retry_after, backoff,
                               attempt + 1, retries)
                time.sleep(backoff)
                continue
            # Other non-200: short backoff
            backoff = min(60, 2 ** attempt)
            logger.warning("  unexpected status=%d body=%r; sleeping %.1fs",
                           r.status_code, r.text[:200], backoff)
            time.sleep(backoff)
        except requests.RequestException as e:
            dt = time.time() - t0
            last_err = e
            backoff = min(60, 2 ** attempt)
            logger.warning("  network error after %.2fs: %s; sleeping %.1fs",
                           dt, e, backoff)
            time.sleep(backoff)
    logger.error("arxiv query exhausted retries (%s) for query=%s",
                 last_err, params["search_query"])
    raise ArxivQueryError(str(last_err))


def find_arxiv_match(title: str, threshold: float = 0.9) -> dict | None:
    """May raise ArxivQueryError; returns None for 'searched, no match'."""
    candidates = arxiv_search_title(title)
    best: tuple[float, str, str] | None = None
    for aid, atitle in candidates:
        sim = title_similarity(title, atitle)
        if best is None or sim > best[0]:
            best = (sim, aid, atitle)
    if best and best[0] >= threshold:
        return {"arxiv_id": best[1], "arxiv_title": best[2], "similarity": best[0]}
    return None


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def sample_papers(name: str, papers: list[dict], seed: int) -> list[dict]:
    """Pick min(100, n) papers; cache the chosen IDs so future runs reuse
    the exact same sample even if the upstream list changes order."""
    sample_path = CACHE_DIR / f"{name}.sample.json"
    if sample_path.exists():
        ids = json.loads(sample_path.read_text())["ids"]
        by_id = {p["id"]: p for p in papers}
        chosen = [by_id[i] for i in ids if i in by_id]
        if len(chosen) == len(ids):
            return chosen
        logger.warning("%s: cached sample contained %d ids, %d still match upstream; "
                       "rebuilding sample.", name, len(ids), len(chosen))

    n = len(papers)
    k = min(100, n)
    rng = random.Random(f"{seed}:{name}")
    chosen = rng.sample(papers, k) if k < n else list(papers)
    atomic_write_json(sample_path, {
        "venue": name,
        "population": n,
        "sample_size": len(chosen),
        "seed": seed,
        "ids": [p["id"] for p in chosen],
    })
    logger.info("%s: sampled %d of %d accepted papers", name, len(chosen), n)
    return chosen


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def load_or_fetch(name: str, fetcher) -> list[dict]:
    path = CACHE_DIR / f"{name}.papers.json"
    if path.exists():
        data = json.loads(path.read_text())
        logger.info("%s: loaded %d accepted papers from cache", name, len(data))
        return data
    logger.info("%s: fetching accepted-paper list ...", name)
    data = fetcher()
    atomic_write_json(path, data)
    logger.info("%s: fetched %d accepted papers", name, len(data))
    return data


def check_coverage(name: str, papers: list[dict], workers: int, sleep: float) -> dict:
    path = CACHE_DIR / f"{name}.arxiv.json"
    matches: dict[str, dict | None] = {}
    if path.exists():
        matches = json.loads(path.read_text())

    pending = [p for p in papers if p["id"] not in matches]
    already = len(papers) - len(pending)
    logger.info("%s: %d/%d already done, %d to process",
                name, already, len(papers), len(pending))

    lock = threading.Lock()
    save_every = 5  # save very frequently for resumability
    counter = {"n": 0}

    def worker(paper: dict) -> tuple[str, dict | None, str | None]:
        if STOP.is_set():
            return paper["id"], None, "stopped"
        logger.debug("%s: querying arxiv for %r (id=%s)",
                     name, paper["title"][:90], paper["id"])
        try:
            result = find_arxiv_match(paper["title"])
            err = None
            if result:
                logger.info("%s: HIT %s (sim=%.2f) <- %r",
                            name, result["arxiv_id"], result["similarity"],
                            paper["title"][:80])
            else:
                logger.info("%s: miss <- %r", name, paper["title"][:80])
        except ArxivQueryError as e:
            result = None
            err = str(e)
        time.sleep(sleep)  # be polite to arXiv (official limit: 1 req / 3s)
        return paper["id"], result, err

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker, p): p for p in pending}
            try:
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=name, unit="paper", file=sys.stderr):
                    paper = futures[fut]
                    try:
                        pid, res, err = fut.result()
                    except Exception as e:
                        logger.exception("%s: worker error on %s: %s",
                                         name, paper["id"], e)
                        continue
                    if err == "stopped":
                        continue
                    if err is not None:
                        # Don't cache failures; they'll be retried on resume.
                        logger.warning("%s: arxiv query failed for %s (%s); "
                                       "will retry on resume",
                                       name, paper["id"][:60], err)
                        continue
                    with lock:
                        matches[pid] = res
                        counter["n"] += 1
                        if counter["n"] % save_every == 0:
                            atomic_write_json(path, matches)
                            logger.info("%s: progress %d done, %d hits so far",
                                        name, len(matches),
                                        sum(1 for m in matches.values() if m))
            finally:
                atomic_write_json(path, matches)

    # Compute coverage over the (sampled) papers only.
    processed = [p for p in papers if p["id"] in matches]
    found = sum(1 for p in processed if matches.get(p["id"]))
    return {
        "total_sampled": len(papers),
        "processed": len(processed),
        "found": found,
        "coverage": (found / len(processed)) if processed else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent arXiv requests (default 1; arXiv allows "
                         "only 1 request per 3 seconds).")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="Per-request delay seconds per worker (default 3.0, "
                         "matching arXiv's official rate limit).")
    ap.add_argument("--seed", type=int, default=20250628,
                    help="Random seed for sampling (default 20250628).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to specific venue labels.")
    ap.add_argument("--no-sample", action="store_true",
                    help="Run the full population instead of sampling (slow).")
    args = ap.parse_args()

    _install_signal_handlers()
    logger.info("=" * 60)
    logger.info("Starting arxiv-coverage run (workers=%d, sleep=%.2fs, seed=%d, "
                "sample=%s)", args.workers, args.sleep, args.seed,
                not args.no_sample)

    venues: dict[str, list[dict]] = {}

    for name, vid in OPENREVIEW_VENUES.items():
        if args.only and name not in args.only:
            continue
        venues[name] = load_or_fetch(name, lambda v=vid: fetch_openreview(v))

    for name, (event, vols) in ANTHOLOGY_VENUES.items():
        if args.only and name not in args.only:
            continue
        venues[name] = load_or_fetch(name, lambda e=event, v=vols: fetch_anthology(e, v))

    # Build per-venue working set (sampled or full).
    work: dict[str, dict] = {}
    for name, papers in venues.items():
        if not papers:
            logger.warning("%s: empty accepted-paper list", name)
            work[name] = {"population": 0, "papers": []}
            continue
        if args.no_sample:
            work[name] = {"population": len(papers), "papers": papers}
        else:
            sampled = sample_papers(name, papers, args.seed)
            work[name] = {"population": len(papers), "papers": sampled}

    summary = {}
    for name, w in work.items():
        if STOP.is_set():
            logger.warning("Stop requested before processing %s", name)
            break
        if not w["papers"]:
            summary[name] = {"population": w["population"], "sampled": 0,
                             "processed": 0, "found": 0, "coverage": 0.0}
            continue
        res = check_coverage(name, w["papers"], args.workers, args.sleep)
        summary[name] = {
            "population": w["population"],
            "sampled": res["total_sampled"],
            "processed": res["processed"],
            "found": res["found"],
            "coverage": res["coverage"],
        }
        atomic_write_json(HERE / "summary.json", summary)
        logger.info("%s: found %d/%d (%.1f%%) on arXiv",
                    name, res["found"], res["processed"],
                    100.0 * res["coverage"])

    # Final print
    print()
    print(f"{'Venue':<14}{'Pop':>8}{'Sample':>8}{'Done':>8}{'Found':>8}{'Cov%':>8}")
    print("-" * 54)
    for name, s in summary.items():
        print(f"{name:<14}{s['population']:>8}{s['sampled']:>8}"
              f"{s['processed']:>8}{s['found']:>8}"
              f"{100*s['coverage']:>7.1f}%")
    atomic_write_json(HERE / "summary.json", summary)
    logger.info("Done. summary.json written.")


if __name__ == "__main__":
    main()
