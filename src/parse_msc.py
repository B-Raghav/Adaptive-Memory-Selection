"""Parse the Multi-Session Chat (MSC) corpus into per-memory records.

A *memory* is a single utterance drawn from a **previous session** of a
multi-session conversation. For a given episode in session ``N`` the memory bank
is every utterance spoken in sessions ``1 .. N-1`` (the ``previous_dialogs``
field). The current session's turns are the *future* against which a memory may
or may not turn out to be relevant.

Two artefacts are written to ``data/processed``:

* ``memories.jsonl`` -- one row per memory with the scalar context needed for
  feature engineering (recency, role, length, the current-session opening query,
  and the human persona-summary annotation used for label validation).
* ``episodes.jsonl`` -- one row per episode holding the future turns and the
  carried-forward persona sentences, consumed by ``labeling.py``.

The persona-worthiness annotation is recovered by matching a memory's text
against the ``msc_personasummary`` files, where each utterance is human-labelled
with the persona fact (if any) it contributes -- an independent signal we use to
validate our automatic labels.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "msc"
OUT = ROOT / "data" / "processed"


def _norm(text: str) -> str:
    """Normalise an utterance for cross-file matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def build_persona_worthiness_lookup() -> dict[str, dict]:
    """Map normalised utterance text -> {persona_worthy, persona_text}.

    Built from every available ``msc_personasummary`` file. In those files each
    turn carries a ``persona_text`` field: a non-empty value means a human judged
    the utterance to contain a durable persona fact worth remembering.
    """
    lookup: dict[str, dict] = {}
    ps_dir = RAW / "msc_personasummary"
    if not ps_dir.exists():
        print("WARNING: msc_personasummary not found; validation labels unavailable.")
        return lookup

    files = sorted(ps_dir.rglob("*.txt"))
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for turn in rec.get("dialog", []):
                    text = turn.get("text", "")
                    if not text:
                        continue
                    persona_text = (turn.get("persona_text") or "").strip()
                    key = _norm(text)
                    # Prefer a positive annotation if the same text recurs.
                    if key not in lookup or (persona_text and not lookup[key]["persona_text"]):
                        lookup[key] = {
                            "persona_worthy": bool(persona_text),
                            "persona_text": persona_text,
                        }
    print(f"Persona-worthiness lookup built from {len(files)} files: {len(lookup)} unique utterances.")
    return lookup


def parse_episodes(sessions: list[int], split: str, max_per_session: int, lookup: dict[str, dict]):
    memories: list[dict] = []
    episodes: list[dict] = []
    mem_id = 0

    for target_session in sessions:
        path = RAW / "msc_dialogue" / f"session_{target_session}" / f"{split}.txt"
        if not path.exists():
            print(f"  skip missing {path}")
            continue

        with open(path, encoding="utf-8") as fh:
            for ep_idx, line in enumerate(fh):
                if ep_idx >= max_per_session:
                    break
                rec = json.loads(line)
                prev = rec.get("previous_dialogs") or []
                if not prev:
                    continue

                cur_dialog = rec.get("dialog") or []
                future_texts = [t.get("text", "") for t in cur_dialog if t.get("text")]
                if not future_texts:
                    continue
                current_query = future_texts[0]
                # Carried-forward persona sentences (both speakers), flattened.
                personas = rec.get("personas") or [[], []]
                persona_sentences = [s for side in personas for s in side]

                episode_id = f"s{target_session}_{split}_{ep_idx}"

                # Flatten previous-session turns in chronological order so we can
                # measure recency in turns up to the recall point (session start).
                flat: list[tuple[int, int, str]] = []  # (memory_session, turn_idx, text)
                for j, pdlg in enumerate(prev):
                    for i, turn in enumerate(pdlg.get("dialog", [])):
                        text = turn.get("text", "")
                        if text:
                            flat.append((j + 1, i, text))
                total_prev = len(flat)

                for global_idx, (memory_session, turn_idx, text) in enumerate(flat):
                    ann = lookup.get(_norm(text), {})
                    memories.append(
                        {
                            "memory_id": mem_id,
                            "episode_id": episode_id,
                            "split": split,
                            "target_session": target_session,
                            "memory_session": memory_session,
                            "recency_sessions": target_session - memory_session,
                            "turns_since_recall": total_prev - global_idx,
                            "position_in_session": turn_idx,
                            "role_speaker1": int(turn_idx % 2 == 0),
                            "text": text,
                            "current_query": current_query,
                            "num_future_turns": len(future_texts),
                            "persona_worthy": ann.get("persona_worthy"),
                            "persona_text": ann.get("persona_text", ""),
                        }
                    )
                    mem_id += 1

                episodes.append(
                    {
                        "episode_id": episode_id,
                        "target_session": target_session,
                        "current_query": current_query,
                        "future_texts": future_texts,
                        "persona_sentences": persona_sentences,
                    }
                )

    return memories, episodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse MSC into memory records.")
    ap.add_argument("--sessions", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--split", default="valid")
    ap.add_argument("--max-per-session", type=int, default=150)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lookup = build_persona_worthiness_lookup()
    memories, episodes = parse_episodes(args.sessions, args.split, args.max_per_session, lookup)

    mem_path = OUT / "memories.jsonl"
    ep_path = OUT / "episodes.jsonl"
    with open(mem_path, "w", encoding="utf-8") as fh:
        for row in memories:
            fh.write(json.dumps(row) + "\n")
    with open(ep_path, "w", encoding="utf-8") as fh:
        for row in episodes:
            fh.write(json.dumps(row) + "\n")

    matched = sum(1 for m in memories if m["persona_worthy"] is not None)
    print(f"Wrote {len(memories)} memories across {len(episodes)} episodes.")
    print(f"  persona-annotation matched: {matched}/{len(memories)} ({100*matched/max(len(memories),1):.1f}%)")
    print(f"  -> {mem_path}")
    print(f"  -> {ep_path}")


if __name__ == "__main__":
    main()
