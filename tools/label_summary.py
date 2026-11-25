"""Summarise labels in a project by author/source/class/state and per frame.  python tools/label_summary.py <project>"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.core.project import Project  # noqa: E402

p = Project(sys.argv[1])
rows = p.db.query("""SELECT a.*, f.path FROM annotations a JOIN frames f ON a.frame_id=f.id WHERE a.deleted=0""")
order = {f["id"]: i for i, f in enumerate(sorted(p.all_frames(), key=lambda f: f["path"]))}
by = Counter()
per_frame = defaultdict(Counter)
geo = Counter()
for r in rows:
    at = json.loads(r["attributes"] or "{}")
    who = "AI-provisional" if at.get("provisional_by") and not at.get("reviewed_by") else (
        "AI-confirmed" if at.get("provisional_by") else r["author"] or "?")
    by[(who, r["class_name"], r["label_state"])] += 1
    per_frame[order[r["frame_id"]]][(r["class_name"], r["label_state"])] += 1
    geo[(r["class_name"], r["geometry_type"])] += 1
print("== by author / class / state")
for k, v in sorted(by.items()):
    print(f"  {k[0]:15s} {k[1]:32s} {k[2]:9s} {v}")
print("== geometry")
for k, v in sorted(geo.items()):
    print(f"  {k[0]:32s} {k[1]:8s} {v}")
print("== per frame (path order)")
for i in sorted(per_frame):
    print(f"  {i:2d}: " + ", ".join(f"{c}/{s[:3]}={n}" for (c, s), n in sorted(per_frame[i].items())))
st = Counter(r["review_status"] for r in p.all_frames())
print("== frame review status", dict(st))
