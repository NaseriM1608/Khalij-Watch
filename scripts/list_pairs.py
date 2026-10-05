"""Write the passes compare_cfar_model.py --pairs compares, by a rule fixed before any result.

For each box: every readable pass in [--since, --until) whose footprint holds
the box centre (build_dataset.passes_over), of the chosen satellites, less
the scenes of the named index splits (by default train and val: scenes the
model was trained or chosen on), then up to --per-area of them spread evenly
in name order (compare_cfar_model.spread; name order is time order within one
satellite). The rule and each box's counts head the file as comments.

Run (CDSE catalogue only, no credentials):
    .venv-eval/Scripts/python scripts/list_pairs.py --since 2024-01-01 --until 2026-01-01 \
        --per-area 12 --out models/runs/kaggle-v2/pairs_iran_s1a_2024-25.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.areas import IRAN_AREAS, STUDY_AREAS, box_around  # noqa: E402
from scripts.compare_cfar_model import spread  # noqa: E402

AREA_SETS = {"iran": IRAN_AREAS, "study": STUDY_AREAS}


def choose(
    found: Mapping[str, Sequence[str]],
    *,
    satellites: Collection[str],
    exclude: Collection[str],
    per_area: int,
) -> tuple[list[tuple[str, str]], dict[str, dict[str, int]]]:
    """(area, scene) pairs to compare, and per box the passes found, of other
    satellites, seen (in ``exclude``), kept and chosen."""
    wanted = {area: {s for s in scenes if s[:3] in satellites} for area, scenes in found.items()}
    kept = {area: sorted(scenes - set(exclude)) for area, scenes in wanted.items()}
    pairs = [(area, s) for area in sorted(kept) for s in kept[area]]
    if per_area:
        pairs = spread(pairs, per_area)
    counts = {
        area: {
            "found": len(found[area]),
            "other_satellites": len(found[area]) - len(wanted[area]),
            "seen": len(wanted[area]) - len(kept[area]),
            "kept": len(kept[area]),
            "chosen": sum(a == area for a, _ in pairs),
        }
        for area in sorted(found)
    }
    return pairs, counts


def seen_scenes(index: Path, splits: Collection[str]) -> set[str]:
    """Scenes of these index splits."""
    scenes = set()
    if not splits:
        return scenes
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] in splits:
                scenes.add(record["scene"])
    return scenes


def write_pairs(path: Path, pairs: Sequence[tuple[str, str]], header: Sequence[str]) -> None:
    """One "area scene" line per pair, after the header as # comments."""
    lines = [f"# {line}" for line in header] + [f"{area} {scene}" for area, scene in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areas", choices=sorted(AREA_SETS), default="iran")
    parser.add_argument("--since", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--until", type=dt.date.fromisoformat, required=True, help="exclusive")
    parser.add_argument("--satellites", default="S1A", help="comma-separated, e.g. S1C,S1D")
    parser.add_argument(
        "--per-area", type=int, default=0, help="evenly spaced passes per box (0 = all)"
    )
    parser.add_argument(
        "--exclude-splits",
        default="train,val",
        help="index splits whose scenes are left out (empty = none)",
    )
    parser.add_argument("--index", type=Path, default=Path("data/datasets/index.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.until <= args.since:
        parser.error("--until must be after --since")

    from scripts.build_dataset import passes_over

    since = dt.datetime.combine(args.since, dt.time(), dt.UTC)
    until = dt.datetime.combine(args.until, dt.time(), dt.UTC)
    satellites = {s.strip() for s in args.satellites.split(",") if s.strip()}
    splits = {s.strip() for s in args.exclude_splits.split(",") if s.strip()}
    found = {
        name: [p.name for p in passes_over(box_around(*spec), since, until)]
        for name, spec in AREA_SETS[args.areas].items()
    }
    pairs, counts = choose(
        found,
        satellites=satellites,
        exclude=seen_scenes(args.index, splits),
        per_area=args.per_area,
    )
    header = [
        f"written {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC by scripts/list_pairs.py",
        f"{args.areas} boxes; passes {args.since}..{args.until} (until exclusive) holding the "
        f"box centre; satellites {','.join(sorted(satellites))}; frames (scenes) of index splits "
        f"{','.join(sorted(splits)) or 'none'} left out (other frames of the same passes kept); "
        f"{f'up to {args.per_area} per box, spread in name order' if args.per_area else 'all'}",
        *(
            f"{area}: found {c['found']}, other satellites {c['other_satellites']}, "
            f"frames of left-out splits {c['seen']}, kept {c['kept']}, chosen {c['chosen']}"
            for area, c in counts.items()
        ),
    ]
    write_pairs(args.out, pairs, header)
    for line in header[2:]:
        print(line, flush=True)
    print(f"{len(pairs)} pairs written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
