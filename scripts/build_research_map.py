#!/usr/bin/env python3
"""Build the research map data (_data/research_map.yml) for /materials/research-map/.

Inputs:
  _i18n/en/nir.md, _i18n/en/thesis.md -- the Research Reports and Theses tables;
  _data/research_topics.yml           -- hand-edited: topics, per-work tags, open problems;
  _data/research_abstracts.yml        -- generated cache: abstracts extracted from the linked
                                         thesis/paper PDFs (or repo READMEs); only works missing
                                         from it are fetched, delete an entry to refetch it.

Each student's rows at one degree level merge into one work. A work is embedded by
Qwen3-Embedding-0.6B as the mean of two views, its titles and its last title + abstract,
laid out in 2D with UMAP weakly guided by the topic tags, and written with the advisor hubs and
the advisor x topic matrix to _data/research_map.yml, rendered by _includes/research_map.html.
Existing points keep their coordinates between runs and new works are placed next to their
nearest neighbours; pass --relayout to recompute the whole layout.
Model and layout were chosen by abstract -> own-work retrieval, trustworthiness and
held-out topic accuracy.

Run locally after adding table rows (CI only renders the committed YAML):
  uv run --python 3.12 --with pyyaml --with pypdf --with sentence-transformers --with umap-learn \
      scripts/build_research_map.py
New works without tags are listed and the script exits with an error:
add them under `tags:` in _data/research_topics.yml and run it again.
"""

from __future__ import annotations

import hashlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
TABLES = {"report": ROOT / "_i18n/en/nir.md", "thesis": ROOT / "_i18n/en/thesis.md"}
TOPICS_FILE = ROOT / "_data/research_topics.yml"
OUT_FILE = ROOT / "_data/research_map.yml"
ABSTRACTS_FILE = ROOT / "_data/research_abstracts.yml"
PDF_CACHE = Path.home() / ".cache/intsystems-research-map"
MODEL = "Qwen/Qwen3-Embedding-0.6B"
PROMPT = "Instruct: Represent the research topic of this student thesis\nQuery: "
WIDTH, PAD = 1000, 40  # SVG viewBox width; the height follows the layout's aspect ratio
LABEL_CHAR_WIDTH = 7.5  # approximate width of one 13px label character, to keep labels inside the frame
HUB_MIN_WORKS = 3  # advisors with fewer works get no hub and no matrix row

# [label](url): the url may hold one level of balanced parentheses ("...nir%20(1).pdf").
LINK = re.compile(r"\[([^\]]*)\]\(<?((?:[^()\s<>]|\([^()\s]*\))*)>?\)")
DEGREE = re.compile(r",?\s*\b(PhD|DSc|Prof\.?|Professor|Ph\.D\.)\b\.?", re.I)
# Maturity level of a work (point size): 1 reports only, 2 defended thesis, 3 preprint, 4 peer-reviewed.
PREPRINT = re.compile(
    r"arxiv\.org|10\.48550/arxiv", re.I
)  # arXiv itself and arXiv DOIs
PEER_REVIEWED = re.compile(
    r"openreview\.net|doi\.org|dl\.acm\.org|springer|ieeexplore|sciencedirect|mdpi|aclanthology"
    r"|proceedings\.mlr\.press|naacl\.org|iospress|crm\.ics\.org\.ru"
)
RADIUS = {1: 3.5, 2: 5.5, 3: 7.5, 4: 9.5}
# Advisor surname spellings -> one key; student first-name and surname variants -> one key.
ADVISOR_ALIAS = {
    "bahteev": "bakhteev",
    "chehovich": "chekhovich",
    "zaitsev": "zaytsev",
    "баранников": "barannikov",
}
ADVISOR_FIRST_LAST = {"lempitsky victor": "lempitsky"}
NAME_ALIAS = {
    "alexandr": "alexander",
    "aleksandr": "alexander",
    "andrei": "andrey",
    "anastasiia": "anastasia",
    "anastasiya": "anastasia",
    "georgii": "georgiy",
    "grigorii": "grigoriy",
    "dmitry": "dmitrii",
    "iurii": "yuriy",
    "alex": "alexey",
    "aleksei": "alexey",
    "gregory": "grigoriy",
    "ilia": "ilya",
    "sergei": "sergey",
    "khristolybov": "khristolyubov",
    "vozniuk": "voznyuk",
    "dementyev": "dementev",
}
# A Latin name with a stray Cyrillic look-alike letter ("Gorсhakov") vs. a name written in Cyrillic.
CYRILLIC_LOOKALIKES = str.maketrans("асеорх", "aceopx")
TRANSLIT = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh shch - y - e yu ya".split(),
    )
)


def plain(cell: str) -> str:
    return re.sub(r"\s+", " ", LINK.sub(r"\1", cell).replace("<br>", " ")).strip()


def advisor_key(name: str) -> str:
    """'Strijov V.V', 'Vadim Strijov', 'Y.V. Dorn' -> 'strijov'."""
    low = name.lower()
    if low in ADVISOR_FIRST_LAST:
        return ADVISOR_FIRST_LAST[low]
    tokens = [t for t in low.split() if "." not in t and len(t) > 1]
    if not tokens:
        return ""
    surname = tokens[0] if re.search(r"\b[a-zа-я]\.", low) else tokens[-1]
    return ADVISOR_ALIAS.get(surname, surname)


def advisors(cell: str) -> list[tuple[str, str]]:
    """Advisor cell -> [(key, name as written)]; adjacent links are separate advisors."""
    cell = DEGREE.sub("", LINK.sub(r"\1;", cell))
    out = []
    for name in re.split(r"<br\s*/?>|;|,|\(|\)|\band\b|\bи\b", cell):
        name = name.strip(" ,.[]()<>")
        if key := advisor_key(name):
            out.append((key, name))
    return out


def latin(token: str) -> str:
    if re.search(r"[a-z]", token):
        return token.translate(CYRILLIC_LOOKALIKES)
    return "".join(TRANSLIT.get(ch, ch) for ch in token).replace("-", "")


def student_key(name: str) -> str:
    tokens = [latin(t) for t in re.findall(r"[a-zа-яё]+", name.lower())]
    return "_".join(sorted(NAME_ALIAS.get(t, t) for t in tokens))


def academic_year(period: str) -> int:
    """'Fall 2025' -> 2026 (the year the academic year ends); 'Spring 2026' and '2026' -> 2026."""
    year = int(period[-4:])
    return year + 1 if period.startswith("Fall") else year


def table_rows(path: Path, kind: str):
    period = level = None
    header: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("### "):
            period = line[4:].strip()
        elif line.startswith("#### "):
            level = line[5:].strip()
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if re.fullmatch(r":?-+:?", cells[0].replace(" ", "")):
                continue
            if cells[0] in ("Student", "PhD Student"):
                header = cells
                continue
            row = dict(zip(header, cells))
            student = plain(row[header[0]])
            if not student:  # the tables have a few empty placeholder rows
                continue
            links = [
                (label.strip(), url.strip())
                for col, cell in row.items()
                if col not in ("Student", "PhD Student", "Advisor", "Title")
                for label, url in LINK.findall(cell)
                if url.startswith("http")
            ]
            yield {
                "kind": kind,
                "year": academic_year(period),
                "term": 2
                if kind == "thesis"
                else 0
                if period.startswith("Fall")
                else 1,
                "stage": "phd"
                if "PhD" in level
                else "master"
                if "Master" in level
                else "bachelor",
                "student": student,
                "title": plain(row["Title"]),
                "advisors": advisors(row["Advisor"]),
                "links": links,
            }


def collect_works() -> tuple[list[dict], dict[str, Counter]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    spellings: dict[str, Counter] = defaultdict(Counter)
    for kind, path in TABLES.items():
        for row in table_rows(path, kind):
            groups[f"{student_key(row['student'])}__{row['stage']}"].append(row)
            for key, name in row["advisors"]:
                spellings[key][name] += 1
    works = []
    for wid, rows in groups.items():
        # Chronological: Fall and Spring of one academic year share `year`, the thesis comes last.
        rows.sort(key=lambda r: (r["year"], r["term"]))
        last = rows[-1]
        advisor_count = Counter(key for r in rows for key, _ in r["advisors"])
        # The latest row (the thesis, if any) has the current links.
        links = dict(last["links"])
        all_urls = [url for r in reversed(rows) for _, url in r["links"]]
        sources = [(label, url) for r in reversed(rows) for label, url in r["links"]]
        works.append(
            {
                "id": wid,
                "student": last["student"],
                "stage": last["stage"],
                "year_from": rows[0]["year"],
                "year": last["year"],
                # Distinct titles, each at its latest position, so titles[-1] is the current one.
                "titles": list(
                    reversed(
                        dict.fromkeys(r["title"] for r in reversed(rows) if r["title"])
                    )
                ),
                "advisors": [key for key, _ in advisor_count.most_common()],
                "level": maturity_level(rows, all_urls),
                "links": [{"label": label, "url": url} for label, url in links.items()],
                "sources": sources,  # all links, newest first; dropped before writing
            }
        )
    works.sort(key=lambda w: (-w["year"], w["student"]))
    return works, spellings


def maturity_level(rows: list[dict], urls: list[str]) -> int:
    if any(PEER_REVIEWED.search(u) and not PREPRINT.search(u) for u in urls):
        return 4
    if any(PREPRINT.search(u) for u in urls):
        return 3
    return 2 if any(r["kind"] == "thesis" for r in rows) else 1


def advisor_name(spellings: Counter) -> str:
    """Prefer a 'First Last' spelling as written in the tables, else the most common one."""
    full = [
        n for n, _ in spellings.most_common() if "." not in n and len(n.split()) == 2
    ]
    if full:
        return full[0]
    name = spellings.most_common(1)[0][0]
    # 'Beznosikov A.N' -> 'Beznosikov A.N.'
    return f"{name}." if re.search(r"\b\w$", name) else name


PAPER_LABEL = re.compile(r"paper|thesis|publication|draft|report|text|arxiv", re.I)
ABSTRACT_START = re.compile(
    r"^\s*(abstract|аннотация|реферат|annotation)\b[\s.:—-]*", re.I | re.M
)
ABSTRACT_STOP = re.compile(
    r"\n\s*(keywords?|key words|index terms|ключевые слова|\d?\.?\s*introduction|введение"
    r"|содержание|оглавление|contents|1\s+[A-ZА-Я])",
    re.I,
)
INTRO_HEADING = re.compile(r"^\s*(?:\d\.?\s*)?(введение|introduction)\s*$", re.I | re.M)
README_SECTION = re.compile(
    r"^#+\s*(abstract|аннотация|описание|description|overview|about)\s*$", re.I | re.M
)


def pdf_url(url: str) -> str | None:
    """Direct PDF URL for the hosts the tables link to, or None."""
    if m := re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+\.pdf)$", url, re.I):
        return f"https://raw.githubusercontent.com/{m[1]}/{m[2]}/{m[3]}"
    if m := re.search(r"arxiv\.org/(?:abs|pdf)/([\w.]+?)(?:v\d+)?(?:\.pdf)?$", url):
        return f"https://arxiv.org/pdf/{m[1]}"
    if m := re.search(r"openreview\.net/(?:forum|pdf)\?id=([\w-]+)", url):
        return f"https://openreview.net/pdf?id={m[1]}"
    if "aclanthology.org" in url:
        return url.rstrip("/") + ("" if url.endswith(".pdf") else ".pdf")
    if m := re.search(r"drive\.google\.com/file/d/([\w-]+)", url):
        return f"https://drive.google.com/uc?export=download&id={m[1]}"
    return None


def http_get(url: str) -> bytes | None:
    # Non-ASCII paths ("/paper/диплом.pdf") are percent-encoded; existing escapes are kept.
    url = urllib.parse.quote(url, safe=":/?&=%#+@,;~!$'()*[]")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        return urllib.request.urlopen(request, timeout=40).read()
    except (
        urllib.error.HTTPError
    ):  # the server answered (404, 403...): there is no such file
        return None
    # Other OSErrors (network down, timeout) propagate: the work is not cached and is retried next run.


def pdf_text(url: str, pages: int = 10) -> str:
    from pypdf import PdfReader

    path = PDF_CACHE / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.pdf"
    if not path.exists():
        data = http_get(url) or b""
        if not data.startswith(
            b"%PDF-"
        ):  # failed or not a PDF: nothing cached, retried next time
            return ""
        path.write_bytes(data)
    try:
        text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages[:pages])
    # pypdf raises many exception types on malformed PDFs; such a PDF is skipped.
    except Exception:
        return ""
    # Some LaTeX theses come out as cp1251 bytes decoded as latin-1.
    if len(re.findall(r"[À-ÿ]", text)) > 0.3 * max(
        1, len(re.findall(r"[A-Za-zÀ-ÿА-Яа-я]", text))
    ):
        try:
            text = text.encode("latin-1").decode("cp1251")
        except UnicodeError:
            pass
    return text


def words(text: str, limit: int, minimum: int) -> str | None:
    text = re.sub(r"-\n(?=[a-zа-я])", "", text)
    tokens = re.sub(r"\s+", " ", text).split()
    while tokens and tokens[-1].isdigit():  # a page number after the abstract
        tokens.pop()
    return " ".join(tokens[:limit]) if len(tokens) >= minimum else None


def readme_section(repo_url: str) -> str | None:
    owner_repo = re.match(r"https://github\.com/([^/]+/[^/#?]+)", repo_url)
    for branch in ("main", "master"):
        raw = http_get(
            f"https://raw.githubusercontent.com/{owner_repo[1]}/{branch}/README.md"
        )
        if raw is None:
            continue
        md = raw.decode("utf-8", "ignore")
        if not (m := README_SECTION.search(md)):
            return None
        body = md[m.end() :]
        body = body[: n.start()] if (n := re.search(r"^#+\s", body, re.M)) else body
        body = re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", " ", body)
        return words(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body), 300, 30)
    return None


def find_abstract(work: dict) -> dict | None:
    """Abstract of the newest paper/thesis PDF; else a repo README section; else the PDF introduction."""
    pdfs = list(
        dict.fromkeys(
            u
            for label, url in work["sources"]
            if PAPER_LABEL.search(label) and (u := pdf_url(url))
        )
    )
    # A PDF still holding template text (lorem ipsum) is skipped; the next candidate is used.
    texts = [
        (url, t)
        for url in pdfs
        if not re.search(r"lorem ipsum", t := pdf_text(url), re.I)
    ]
    for url, text in texts:
        if m := ABSTRACT_START.search(text[:20000]):
            body = text[m.end() :]
            body = (
                body[: s.start()] if (s := ABSTRACT_STOP.search(body)) else body[:2500]
            )
            if abstract := words(body, 300, 30):
                return {"source": url, "kind": "abstract", "text": abstract}
    repos = dict.fromkeys(
        m[0]
        for _, url in work["sources"]
        if (m := re.match(r"https://github\.com/[^/]+/[^/#?]+", url))
    )
    for repo in list(repos)[:3]:
        if section := readme_section(repo):
            return {"source": repo, "kind": "abstract", "text": section}
    for url, text in texts:
        if headings := list(INTRO_HEADING.finditer(text)):
            if intro := words(text[headings[-1].end() :], 200, 60):
                return {"source": url, "kind": "introduction", "text": intro}
    return None


def load_abstracts(works: list[dict]) -> dict[str, dict]:
    """Cached abstracts; fetches works missing from the cache (a null entry = nothing found).

    A work whose fetch hit a network error or timeout is left out of the cache, so it is retried."""
    cache = (
        yaml.safe_load(ABSTRACTS_FILE.read_text()) if ABSTRACTS_FILE.exists() else {}
    )
    ids = {w["id"] for w in works}
    stale = [k for k in cache if k not in ids]  # works that were renamed or merged
    todo = [w for w in works if w["id"] not in cache]
    for key in stale:
        del cache[key]
    if todo or stale:
        PDF_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Fetching abstracts for {len(todo)} works...")
        with ThreadPoolExecutor(8) as pool:
            futures = {w["id"]: pool.submit(find_abstract, w) for w in todo}
        failed = [
            wid for wid, f in futures.items() if isinstance(f.exception(), OSError)
        ]
        cache.update(
            {wid: f.result() for wid, f in futures.items() if wid not in failed}
        )
        if failed:
            print(
                f"{len(failed)} works hit network errors and will be retried next run: {failed}"
            )
        header = "# Generated by scripts/build_research_map.py: abstracts of the linked PDFs. Delete an entry to refetch it.\n"
        ABSTRACTS_FILE.write_text(
            header
            + yaml.safe_dump(
                dict(sorted(cache.items())), allow_unicode=True, width=1000
            )
        )
    return cache


def embed(works: list[dict], abstracts: dict[str, dict]) -> np.ndarray:
    """Mean of two normalized views: all titles, and the last title + abstract (titles only if none)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL)
    model.max_seq_length = 512

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            [PROMPT + t for t in texts], normalize_embeddings=True, batch_size=16
        )

    titles = encode([" ; ".join(w["titles"]) for w in works])
    full = encode(
        [
            f"{w['titles'][-1]}. {a['text']}"
            if (a := abstracts.get(w["id"]))
            else " ; ".join(w["titles"])
            for w in works
        ]
    )
    mixed = titles + full
    return mixed / np.linalg.norm(mixed, axis=1, keepdims=True)


def place_new(
    works: list[dict], vectors: np.ndarray, old: dict[str, tuple[float, float]]
) -> np.ndarray:
    """Keep the previous coordinates; put each new work at the similarity-weighted mean of its
    5 nearest (by embedding) placed works. UMAP reshuffles whole clusters when a few points
    are added, so a full layout only runs on --relayout (or when there is no previous map)."""
    placed = np.array([w["id"] in old for w in works])
    xy = np.array([old.get(w["id"], (0.0, 0.0)) for w in works], dtype=float)
    sim = vectors[~placed] @ vectors[placed].T
    for row, i in enumerate(np.where(~placed)[0]):
        top = np.argsort(-sim[row])[:5]
        weights = np.maximum(sim[row, top], 0) ** 4
        xy[i] = (xy[placed][top] * weights[:, None]).sum(0) / weights.sum()
    return xy


def layout(vectors: np.ndarray, primary: list[str]) -> tuple[np.ndarray, float]:
    """Full 2D layout in viewBox units and the viewBox height."""
    import umap

    codes = {t: i for i, t in enumerate(sorted(set(primary)))}
    xy = umap.UMAP(
        n_neighbors=12,
        min_dist=1.0,
        spread=3.0,  # evens out the gaps between clusters (space usage 28% -> 47%, same trustworthiness)
        metric="cosine",
        target_weight=0.2,
        random_state=0,
    ).fit_transform(vectors, y=[codes[t] for t in primary])
    xy -= xy.min(0)
    scale = (WIDTH - 2 * PAD) / xy[:, 0].max()
    height = float(np.clip(xy[:, 1].max() * scale + 2 * PAD, 0.5 * WIDTH, 0.75 * WIDTH))
    xy[:, 1] *= (height - 2 * PAD) / xy[:, 1].max()  # a mild vertical stretch at most
    xy[:, 0] *= scale
    return xy + PAD, height


def medoid(points: np.ndarray) -> np.ndarray:
    """The point closest to the dense half of its cluster: a label anchor that ignores outliers."""
    d = ((points[:, None] - points[None]) ** 2).sum(-1)
    return points[np.argmin(np.sort(d, 1)[:, : max(3, len(points) // 2)].sum(1))]


def main() -> None:
    config = yaml.safe_load(TOPICS_FILE.read_text())
    directions, topics, tags = config["directions"], config["topics"], config["tags"]
    works, spellings = collect_works()

    untagged = [w for w in works if w["id"] not in tags]
    if untagged:
        print(
            f"{len(untagged)} works have no tags; add them under `tags:` in {TOPICS_FILE.relative_to(ROOT)}:"
        )
        for w in untagged:
            print(f"  {w['id']}: []  # {w['year']} {w['student']}: {w['titles'][-1]}")
        sys.exit(1)
    unknown = {t for wid in tags for t in tags[wid] if t not in topics}
    if unknown:
        sys.exit(f"Unknown topics in tags: {sorted(unknown)}")
    for w in works:
        w["topics"] = tags[w["id"]]

    abstracts = load_abstracts(works)
    vectors = embed(works, abstracts)
    previous = yaml.safe_load(OUT_FILE.read_text()) if OUT_FILE.exists() else None
    old = {w["id"]: (w["x"], w["y"]) for w in previous["works"]} if previous else {}
    kept = sum(w["id"] in old for w in works)
    if "--relayout" not in sys.argv and kept >= 10:
        xy, height = place_new(works, vectors, old), previous["height"]
        print(f"Kept {kept} points, placed the new ones (--relayout to redo all)")
    else:
        xy, height = layout(vectors, [w["topics"][0] for w in works])
    for w, (x, y) in zip(works, xy):
        w.update(x=round(float(x), 1), y=round(float(y), 1), r=RADIUS[w["level"]])
        w["title"] = w.pop("titles")[-1]
        del w["sources"]
        if (a := abstracts.get(w["id"])) and a["kind"] == "abstract":
            w["abstract"] = a["text"]

    work_count = Counter(a for w in works for a in w["advisors"])
    hubs = []
    for key, n in work_count.most_common():
        if n < HUB_MIN_WORKS:
            break
        pts = np.array([(w["x"], w["y"]) for w in works if key in w["advisors"]])
        x, y = pts.mean(0)
        profile = next((p.stem for p in (ROOT / "_people").glob(f"{key}_*.md")), None)
        hubs.append(
            {
                "id": key,
                "name": advisor_name(spellings[key]),
                "surname": key.capitalize(),
                "profile": profile,
                "count": n,
                "x": round(float(x), 1),
                "y": round(float(y), 1),
            }
        )
    hub_ids = {h["id"] for h in hubs}
    names = {key: advisor_name(s) for key, s in spellings.items()}
    for w in works:
        w["advisor_names"] = [names[a] for a in w["advisors"]]

    topic_list = []
    for tid, t in topics.items():
        members = [w for w in works if tid in w["topics"]]
        if not members:
            continue
        core = np.array(
            [(w["x"], w["y"]) for w in members if w["topics"][0] == tid]
            or [(w["x"], w["y"]) for w in members]
        )
        x, y = medoid(core) if len(core) >= 3 else core.mean(0)
        half = LABEL_CHAR_WIDTH * max(len(t["en"]), len(t["ru"])) / 2
        x = min(max(x, half + 4), WIDTH - half - 4)
        topic_list.append(
            {
                "id": tid,
                "direction": t.get("direction"),
                "en": t["en"],
                "ru": t["ru"],
                "count": len(members),
                "open_problems": t.get("open_problems"),
                "x": round(float(x), 1),
                "y": round(float(y), 1),
            }
        )
    # Grouped by direction (in file order), larger topics first; topics without a direction last.
    order = {d: i for i, d in enumerate(directions)}
    topic_list.sort(key=lambda t: (order.get(t["direction"], len(order)), -t["count"]))
    direction_list = [
        {
            "id": d,
            "en": v["en"],
            "ru": v["ru"],
            "topics": [t["id"] for t in topic_list if t["direction"] == d],
            "count": sum(
                1
                for w in works
                if any(topics[t].get("direction") == d for t in w["topics"])
            ),
        }
        for d, v in directions.items()
    ]

    peak = max(
        sum(1 for w in works if h["id"] in w["advisors"] and t["id"] in w["topics"])
        for h in hubs
        for t in topic_list
    )
    matrix = []
    for h in hubs:
        cells = []
        for t in topic_list:
            n = sum(
                1 for w in works if h["id"] in w["advisors"] and t["id"] in w["topics"]
            )
            cells.append(
                {
                    "n": n,
                    "level": 0 if n == 0 else 1 + min(3, int(3 * (n / peak) ** 0.5)),
                }
            )
        matrix.append({"advisor": h["id"], "cells": cells})

    data = {
        "width": WIDTH,
        "height": round(height),
        "years": f"{min(w['year_from'] for w in works)}–{max(w['year'] for w in works)}",
        "levels": [{"level": k, "r": r} for k, r in RADIUS.items()],
        "directions": direction_list,
        "topics": topic_list,
        "advisors": hubs,
        "edges": [
            {
                "advisor": a,
                "work": w["id"],
                "x1": h["x"],
                "y1": h["y"],
                "x2": w["x"],
                "y2": w["y"],
            }
            for w in works
            for a in w["advisors"]
            if a in hub_ids
            for h in hubs
            if h["id"] == a
        ],
        "works": works,
        "matrix": matrix,
    }
    header = "# Generated by scripts/build_research_map.py; do not edit. Tags and topics: _data/research_topics.yml\n"
    OUT_FILE.write_text(
        header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)
    )
    print(
        f"{len(works)} works, {len(hubs)} advisor hubs, {len(topic_list)} topics -> {OUT_FILE.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
