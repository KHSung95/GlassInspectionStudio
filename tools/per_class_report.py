"""Markdown report from tools/per_class_sweep.py output.  python tools/per_class_report.py <per_class_dir> <out.md>"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
s = json.loads((d / "per_class_summary.json").read_text(encoding="utf-8"))
names = {"black_spot": "흑점 (Black Spot)", "bubble": "기포 (Bubble/Open/Closed)",
         "intensity_tint": "명암·착색 (Intensity/Transparency, Faint Tint)",
         "linear_void": "선형·부메랑 파임 (Linear/Broken Void, Other Defect)"}
L = ["# 결함 종류별 고전 CV 평가 (사용자 라벨 기준)", "",
     "- recall = 라벨된 결함 중 검출된 비율 (점: 허용오차 내, 박스: 박스 안/주변)",
     "- FPver = 사용자가 정상으로 표시한 영역의 검출 수, fa/MP = 처리 영역 1 MP당 (정상 확인 + 미확인) 후보 수",
     "  - 라벨 주변 crop만 처리하므로 fa/MP는 구조물이 많은 영역 기준이며 전체 프레임보다 높게 나올 수 있음", ""]
for g, algs in s.items():
    L += [f"## {names.get(g, g)}", "", "| 검출기 | 설정 | threshold | recall | FPver | fa/MP | 주요 파라미터 |",
          "|---|---|---|---|---|---|---|"]
    for alg, r in algs.items():
        if "error" in r:
            L.append(f"| {alg} | 오류 | | | | | {r['error'][:80]} |"); continue
        b = r["baseline"]
        L.append(f"| {alg} (GT {r['n_gt']}) | 기본값 | {b['threshold']} | {b['recall']} | {b['fp_verified']} | "
                 f"{round(b['false_alarms_per_mp'] or 0, 1)} | - |")
        for x in r["recommendations"]:
            m = x["tune"]
            pr = {k: v for k, v in x["params"].items() if k not in ("threshold", "max_candidates", "max_area", "pre_sigma")}
            L.append(f"| | {x['objective']} | {m['threshold']} | {m['recall']} | {m['fp_verified']} | "
                     f"{round(m['false_alarms_per_mp'] or 0, 1)} | `{json.dumps(pr)}` |")
        for w in r["warnings"]:
            L.append(f"| | ⚠ | | | | | {w[:110]} |")
    L.append("")
Path(sys.argv[2]).write_text("\n".join(L), encoding="utf-8")
print(sys.argv[2])
