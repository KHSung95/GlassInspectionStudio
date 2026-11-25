"""Self-contained HTML report of a project: data status, labelled examples per class, per-class evaluation, limits.

    python tools/studio_report.py <project> <out.html> [--title "..."]
Images are embedded (base64) so the single file can be shared.
"""
import argparse
import base64
import html
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gistudio.core.annotations import AnnotationStore  # noqa: E402
from gistudio.core.geometry import geometry_bbox, geometry_center  # noqa: E402
from gistudio.core.image_io import load_mono  # noqa: E402
from gistudio.core.project import Project  # noqa: E402

GNAME = {"black_spot": "흑점", "bubble": "기포", "intensity_tint": "명암·착색", "linear_void": "선형·부메랑 파임"}
OBJ = {"balanced": "균형", "high_recall": "높은 검출률", "low_fp": "낮은 오검출", "zero_verified_fp": "정상영역 오검출 0",
       "review_budget": "검토 부담 제한"}


def b64png(img) -> str:
    ok, enc = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()


def crop_tile(img, a, size=180):
    x0, y0, x1, y1 = geometry_bbox(a["geometry_type"], a["geometry"])
    cx, cy = geometry_center(a["geometry_type"], a["geometry"])
    w, h = (6, 6) if a["geometry_type"] == "point" else (max(4, x1 - x0), max(4, y1 - y0))
    s = int(max(64, min(500, 3 * max(w, h))))
    H, W = img.shape
    X0, Y0 = int(np.clip(cx - s / 2, 0, max(0, W - s))), int(np.clip(cy - s / 2, 0, max(0, H - s)))
    c = img[Y0:Y0 + s, X0:X0 + s]
    lo, hi = np.percentile(c, (1, 99.5))
    d = np.clip((c.astype(np.float32) - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
    t = cv2.cvtColor(cv2.resize(d, (size, size), interpolation=cv2.INTER_NEAREST if s <= size else cv2.INTER_AREA),
                     cv2.COLOR_GRAY2BGR)
    k = size / s
    if a["geometry_type"] == "point":
        cv2.circle(t, (int((cx - X0) * k), int((cy - Y0) * k)), 10, (0, 0, 255), 1)
    else:
        cv2.rectangle(t, (int((x0 - X0) * k), int((y0 - Y0) * k)), (int((x1 - X0) * k), int((y1 - Y0) * k)), (0, 0, 255), 1)
    return t, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project"); ap.add_argument("out"); ap.add_argument("--title", default="")
    ap.add_argument("--max-per-class", type=int, default=12)
    a = ap.parse_args()
    p = Project(a.project)
    store = AnnotationStore(p)
    st = p.stats()
    frames = sorted(p.all_frames(), key=lambda f: f["path"])
    pos = {f["id"]: i for i, f in enumerate(frames)}
    title = a.title or f"{p.name} 검사 데이터 보고서"
    E = html.escape
    H = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{E(title)}</title><style>"
         "body{font:14px/1.55 'Malgun Gothic',system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2228}"
         "h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}"
         "table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}"
         "th{background:#f4f6f8}.muted{color:#6b7280}.warn{color:#b45309}.tiles{display:flex;flex-wrap:wrap;gap:6px}"
         ".tiles figure{margin:0;font-size:11px;color:#6b7280;width:180px}.tiles img{display:block;border:1px solid #ccc}"
         "</style></head><body>",
         f"<h1>{E(title)}</h1><p class='muted'>생성: {datetime.now():%Y-%m-%d %H:%M} · 프로젝트: {E(str(p.root))}</p>"]
    # ---------------- data status
    H.append("<h2>1. 데이터 현황</h2><table><tr><th>병</th><th>세션</th><th>프레임</th><th>라벨</th><th>결함 라벨</th>"
             f"<th>정상 확인 병</th></tr><tr><td>{st['bottles']}</td><td>{st['sessions']}</td><td>{st['frames']}</td>"
             f"<td>{st['annotations']}</td><td>{st['confirmed_positive']}</td><td>{st['ok_bottles']}</td></tr></table>")
    H.append("<table><tr><th>병</th><th>세션</th><th>프레임</th><th>카메라</th><th>비트</th><th>노출(µs)</th><th>원본 위치</th></tr>")
    for b in p.bottles():
        for s in p.sessions(b["id"]):
            n = p.db.one("SELECT COUNT(*), MIN(width), MIN(height), MIN(dtype) FROM frames WHERE session_id=?", (s["id"],))
            H.append(f"<tr><td>{E(b['bottle_code'])}</td><td>{E(s['session_code'])}</td><td>{n[0]} ({n[1]}×{n[2]} {n[3]})</td>"
                     f"<td>{E(s['camera_model'] or '-')}</td><td>{s['bit_depth'] or '-'}</td><td>{s['exposure_us'] or '-'}</td>"
                     f"<td class='muted'>{E(s['source_dir'] or '')}</td></tr>")
    H.append("</table>")
    rs = dict(p.db.query("SELECT review_status, COUNT(*) FROM frames GROUP BY review_status"))
    H.append(f"<p>프레임 검토 상태: " + ", ".join(f"{k} {v}" for k, v in rs.items()) + "</p>")
    # ---------------- labels
    H.append("<h2>2. 라벨 현황</h2><table><tr><th>클래스</th><th>상태</th><th>개수</th><th>작성</th></tr>")
    rows = p.db.query("""SELECT class_name, label_state, author, COUNT(*) n FROM annotations WHERE deleted=0
                         GROUP BY class_name, label_state, author ORDER BY class_name""")
    for r in rows:
        H.append(f"<tr><td>{E(r['class_name'])}</td><td>{E(r['label_state'])}</td><td>{r['n']}</td><td>{E(r['author'] or '-')}</td></tr>")
    H.append("</table>")
    # examples
    groups = defaultdict(list)
    for f in frames:
        for an in store.for_frame(f["id"], confirmed_only=True):
            if an["label_state"] == "positive":
                groups[an["class_name"]].append((f, an))
    H.append("<h2>3. 결함 예시 (원본 해상도 crop, 화면용 대비 보정)</h2>")
    for cls, items in groups.items():
        H.append(f"<h3>{E(cls)} <span class='muted'>({len(items)}개)</span></h3><div class='tiles'>")
        for f, an in items[: a.max_per_class]:
            t, s = crop_tile(load_mono(f["path"]), an)
            H.append(f"<figure><img src='{b64png(t)}' width='180' height='180'><figcaption>프레임 {pos[f['id']]} · "
                     f"{E(an['geometry_type'])} · 시야 {s}px</figcaption></figure>")
        H.append("</div>")
    # ---------------- evaluation
    pc = p.root / "exports" / "per_class" / "per_class_summary.json"
    H.append("<h2>4. 결함 종류별 고전 CV 평가</h2>")
    if pc.exists():
        summ = json.loads(pc.read_text(encoding="utf-8"))
        H.append(f"<p class='muted'>평가 시각: {datetime.fromtimestamp(pc.stat().st_mtime):%Y-%m-%d %H:%M} · 검출률 = 라벨된 결함 중 검출 비율 · "
                 "1MP당 오검출 후보 = 라벨 주변 처리 영역에서 결함이 아닌 후보 밀도(정상 확인 + 미확인)</p>")
        for g, algs in summ.items():
            H.append(f"<h3>{GNAME.get(g, g)}</h3><table><tr><th>검출기</th><th>설정</th><th>threshold</th><th>검출률</th>"
                     "<th>정상영역 오검출</th><th>1MP당 오검출 후보</th></tr>")
            for alg, x in algs.items():
                if "error" in x:
                    H.append(f"<tr><td>{E(alg)}</td><td colspan='5'>{E(x['error'])}</td></tr>"); continue
                b = x["baseline"]
                H.append(f"<tr><td>{E(alg.replace('classical.', ''))} (GT {x['n_gt']})</td><td>기본값</td><td>{b['threshold']}</td>"
                         f"<td>{b['recall']}</td><td>{b['fp_verified']}</td><td>{round(b['false_alarms_per_mp'] or 0)}</td></tr>")
                for o in x["recommendations"]:
                    m = o["tune"]
                    H.append(f"<tr><td></td><td>{OBJ.get(o['objective'], o['objective'])}</td><td>{m['threshold']}</td>"
                             f"<td><b>{m['recall']}</b></td><td>{m['fp_verified']}</td><td>{round(m['false_alarms_per_mp'] or 0)}</td></tr>")
                for w in x.get("warnings", []):
                    H.append(f"<tr><td></td><td colspan='5' class='warn'>{E(w)}</td></tr>")
            H.append("</table>")
    else:
        H.append("<p class='muted'>아직 평가를 실행하지 않았습니다 (스튜디오 → 분석 → 결함 종류별 평가).</p>")
    # ---------------- limits
    H.append("<h2>5. 해석 시 주의</h2><ul>")
    if not st["ok_bottles"]:
        H.append("<li class='warn'>정상 확인 병이 없어 실제 오배출률은 측정할 수 없습니다.</li>")
    for cls, items in groups.items():
        if len(items) < 10:
            H.append(f"<li class='warn'>{E(cls)} 라벨이 {len(items)}개뿐이라 수치가 불안정하고 낙관적일 수 있습니다.</li>")
    neg = p.db.one("SELECT COUNT(*) FROM annotations WHERE deleted=0 AND label_state='negative'")[0]
    if neg < 20:
        H.append(f"<li class='warn'>정상 영역 라벨이 {neg}개뿐이라 오검출을 대부분 '미확인'으로만 셀 수 있습니다.</li>")
    H.append("<li>표시하지 않은 영역은 Unknown으로 취급되며, 그 안의 검출은 오검출로 세지 않습니다.</li>"
             "<li>추천 파라미터는 현재 라벨에 맞춘 값으로, 다른 병·조명에서 검증이 필요합니다.</li></ul>")
    H.append("</body></html>")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(H), encoding="utf-8")
    print(a.out)


if __name__ == "__main__":
    main()
