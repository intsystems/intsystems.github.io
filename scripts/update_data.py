#!/usr/bin/env python3
"""Refresh externally sourced site data (run weekly by .github/workflows/refresh-data.yml).

1. _data/conferences.yml — deadlines, dates and venues of tracked conferences
   from the community-maintained ccfddl/ccf-deadlines repository. Only entries
   with a `ccfddl:` key (path inside its `conference/` folder, e.g. AI/nips)
   are updated; everything else in the file is left as is.
2. _data/stats.yml — numbers shown on the home page (public repositories of
   the GitHub organization).

Usage: python3 scripts/update_data.py   (needs PyYAML; GITHUB_TOKEN optional)
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CCFDDL_RAW = "https://raw.githubusercontent.com/ccfddl/ccf-deadlines/main/conference/{}.yml"
GITHUB_ORG_API = "https://api.github.com/orgs/{}"

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def fetch(url: str) -> str:
    headers = {"User-Agent": "intsystems.github.io data refresh"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
        return r.read().decode("utf-8")


def parse_deadline(value) -> date | None:
    """ccfddl deadlines look like '2025-05-15 23:59:59' or 'TBD'."""
    if not value or str(value).upper() == "TBD":
        return None
    match = re.match(r"(\d{4}-\d{2}-\d{2})", str(value))
    return datetime.strptime(match.group(1), "%Y-%m-%d").date() if match else None


def parse_conference_start(text: str, year: int) -> str | None:
    """'December 9-15, 2025' / 'Nov 28-Dec 9, 2022' -> '2025-12-09'."""
    if not text:
        return None
    match = re.match(r"\s*([A-Za-z]+)\.?\s+(\d{1,2})", text)
    if not match:
        return None
    month = MONTHS.get(match.group(1).lower()) or MONTHS.get(
        next((m for m in MONTHS if m.startswith(match.group(1).lower()[:3])), ""), None)
    if not month:
        return None
    year_match = re.search(r"(\d{4})\s*$", text)
    return date(int(year_match.group(1)) if year_match else year, month, int(match.group(2))).isoformat()


def main_track(edition: dict) -> dict | None:
    """First timeline entry with a real (non-TBD) deadline — the main track."""
    for entry in edition.get("timeline") or []:
        if parse_deadline(entry.get("deadline")):
            return entry
    return None


def pick_edition(editions: list[dict], today: date) -> dict | None:
    """Earliest edition whose main deadline is still ahead; otherwise the latest one."""
    editions = sorted(editions, key=lambda e: e.get("year", 0))
    for edition in editions:
        track = main_track(edition)
        if track and parse_deadline(track["deadline"]) >= today:
            return edition
    return editions[-1] if editions else None


def update_conferences(today: date) -> bool:
    path = ROOT / "_data" / "conferences.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = False
    for conf in data["conferences"]:
        key = conf.get("ccfddl")
        if not key:
            continue
        try:
            feed = yaml.safe_load(fetch(CCFDDL_RAW.format(key)))[0]
        except Exception as exc:  # network / schema hiccup: keep current values
            print(f"  ! {conf['name']}: could not fetch {key}: {exc}", file=sys.stderr)
            continue
        edition = pick_edition(feed.get("confs") or [], today)
        track = main_track(edition) if edition else None
        if not edition or not track:
            print(f"  ! {conf['name']}: no usable edition in feed", file=sys.stderr)
            continue
        before = dict(conf)
        conf["submission_deadline"] = parse_deadline(track["deadline"]).isoformat()
        abstract = parse_deadline(track.get("abstract_deadline"))
        if abstract:
            conf["abstract_deadline"] = abstract.isoformat()
        else:
            conf.pop("abstract_deadline", None)
        start = parse_conference_start(edition.get("date", ""), edition.get("year"))
        if start:
            conf["conference_date"] = start
        if edition.get("place"):
            conf["location"] = edition["place"]
        if edition.get("link"):
            conf["website"] = edition["link"]
        if conf != before:
            changed = True
            print(f"  {conf['name']}: {edition.get('year')} — deadline {conf['submission_deadline']}")
    if changed:
        header = ("# Conference deadlines shown on /materials/conferences/.\n"
                  "# Entries with a `ccfddl:` key are refreshed weekly by scripts/update_data.py\n"
                  "# from https://github.com/ccfddl/ccf-deadlines — edit only name/full_name/rank/active\n"
                  "# for those; entries without `ccfddl:` are maintained by hand.\n")
        path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=1000),
                        encoding="utf-8")
    return changed


def update_stats(today: date) -> bool:
    path = ROOT / "_data" / "stats.yml"
    config = yaml.safe_load((ROOT / "_config.yml").read_text(encoding="utf-8"))
    org = json.loads(fetch(GITHUB_ORG_API.format(config["github"])))
    current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    stats = {"github_repos": org["public_repos"], "updated": today.isoformat()}
    if current.get("github_repos") == stats["github_repos"]:
        return False
    path.write_text("# Auto-updated weekly by scripts/update_data.py; used on the home page.\n"
                    + yaml.safe_dump(stats, sort_keys=False), encoding="utf-8")
    print(f"  GitHub public repos: {stats['github_repos']}")
    return True


if __name__ == "__main__":
    today = date.today()
    print("Conferences:")
    changed = update_conferences(today)
    print("Stats:")
    changed = update_stats(today) or changed
    print("changed" if changed else "no changes")
