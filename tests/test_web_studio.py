"""Web studio core: open/create project, DB backup, project info, background jobs, report generation."""
import time
from pathlib import Path

import numpy as np
import tifffile

from gistudio.app import web_studio
from gistudio.app.web_studio import Studio, backup_db


def _wait(st, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = next(x for x in st.jobs.items if x["id"] == jid)
        if j["status"] != "running":
            return j
        time.sleep(0.5)
    raise TimeoutError


def test_studio_project_import_report(tmp_path, monkeypatch):
    monkeypatch.setattr(web_studio, "SETTINGS_FILE", tmp_path / "studio.json")
    st = Studio()
    info = st.open_project(str(tmp_path / "proj"), create=True)
    assert info["open"] and Path(info["last_backup"]).exists()
    assert st.recent() == [info["root"]]
    raw = tmp_path / "raw"; raw.mkdir()
    for i in range(2):
        tifffile.imwrite(raw / f"f_a{i * 90}.tif", np.full((40, 60), 100 + i, np.uint8))
    job = st.job_import({"path": str(raw), "layout": "flat", "bottle": "B1", "dataset": "t"})
    j = _wait(st, job["id"])
    assert j["status"] == "done", Path(j["log"]).read_text(encoding="utf-8")
    info = st.project_info()
    assert info["stats"]["frames"] == 2 and info["sessions"][0]["bottle"] == "B1"
    s = st.save_project_settings({"camera": {"packet_size": 8192, "fps": 5.0, "cti": None}})
    assert st.project_settings()["camera"]["fps"] == 5.0 and s["camera"]["packet_size"] == 8192
    job = st.job_report({"title": "t"})
    j = _wait(st, job["id"])
    assert j["status"] == "done", Path(j["log"]).read_text(encoding="utf-8")
    reps = st.reports()
    assert reps and reps[0]["name"].startswith("report_")
    html = (st.project.root / reps[0]["rel"]).read_text(encoding="utf-8")
    assert "데이터 현황" in html and "정상 확인 병이 없어" in html
    for _ in range(12):
        backup_db(st.project, keep=10)
    assert len(list((st.project.root / "backups").glob("studio_*.db"))) <= 10
