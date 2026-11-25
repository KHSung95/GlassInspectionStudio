# 데이터 포맷 명세 (gistudio-project/1)

## 프로젝트 폴더

```
<project>/
  project.json            {name, created_at, format: "gistudio-project/1"}
  studio.db               SQLite (WAL). 모든 메타데이터·라벨·후보·실험 기록
  masks/<uuid>.npz        Brush mask (uint8, key "mask"); geometry {x0,y0,w,h}와 함께 사용
  cache/thumbs/           썸네일 캐시 (삭제해도 됨)
  cache/targets/          학습 타깃 캐시 (라벨 변경 시 자동 무효화, 삭제해도 됨)
  runs/<run_uid>/         run.json, summary.json, maps_<frame_id>.npz (float16 점수맵),
                          config.json, metrics.jsonl, best.pt, last.pt, tb/
  presets/*.json          {algorithm, params}
  prompts/*.json          {normal:[...], defect:[...]}
  exports/                평가 리포트, NG 근거 이미지, 라벨 export
```

원본 이미지는 프로젝트 밖에 그대로 두고 `frames.path`(절대경로)와 `frames.sha256`으로 참조합니다.

## 좌표계

* 단위는 원본 이미지의 연속 픽셀 좌표입니다. x는 오른쪽, y는 아래쪽으로 증가합니다.
* 픽셀 (i, j)는 영역 [i, i+1) × [j, j+1)을 차지하고, 중심은 (i+0.5, j+0.5)입니다.
* 화면 확대나 표시 좌표는 저장하지 않습니다.

## 테이블 (주요 컬럼)

| 테이블 | 설명 |
|---|---|
| `datasets` | id, name |
| `bottles` | bottle_code, product_type, variety, lot_no, physical_sample_id(Split 기준), verdict(unknown/ng/ok/under_review), memo |
| `sessions` | session_code, captured_at, camera_model, bit_depth(유효 비트), exposure_us, gain_db, lighting, lens, magnification, mm_per_px, source_dir |
| `frames` | frame_index, angle_deg, path, sha256, file_size, width, height, dtype, container_bits, review_status |
| `classes` | name, kind(defect/normal/appearance/meta), color, shortcut, train_channel(black_spot/bubble/intensity/'') |
| `defects` | 물리 결함: defect_uid, class_name, final_verdict(undecided/ng/ok), open_state(open/closed/undetermined), keyframe_id |
| `annotations` | frame_id, defect_id, class_name, label_state(positive/negative/unknown/ignore), geometry_type, geometry(JSON), mask_path, visibility(visible/weak/not_visible/uncertain), is_keyframe, source(manual/copied/interpolated/tracked/ai_accepted/ai_rejected/imported), **confirmed**(0이면 GT 제외), verdict, attributes(JSON), deleted |
| `annotation_history` | annotation_id, action(create/update/delete/undo-*/redo-*), before, after, author, ts |
| `review_scopes` | (frame_id, class_name): 해당 클래스는 이 프레임에서 전수 검토됨 |
| `runs` | kind(classical/zeroshot/infer/train/eval), algorithm, params, preprocess, model_info, score_semantics, status, output_dir |
| `candidates` | run_id, frame_id, model_id, source_algorithm, class_name, score, geometry_type, geometry, area_px, inference_ms, group_key, review_status(pending/accepted/rejected/unknown), linked_annotation_id, features |
| `splits`, `split_assignments` | bottle → train/val/test/excluded |

## Geometry JSON

| type | 형식 |
|---|---|
| point | `{"x": float, "y": float, "r": float}`. `r`은 표시 반경이며 물리 크기가 아님 |
| bbox | `{"x0","y0","x1","y1"}` |
| polygon | `{"points": [[x,y], ...]}` |
| mask | `{"x0","y0","w","h"}` + `mask_path` |

## 학습 타깃 규칙

`core/label_targets.py` 상단 docstring에 정의되어 있습니다.

* 기본값: 모든 픽셀이 valid=False(Unknown)
* 우선순위: Review scope → Negative → Positive → Unknown·미결 외관 → Ignore (뒤의 규칙이 앞의 규칙을 덮어씀)

## Export

**File → Export annotations**는 `gistudio-annotations/1` JSON을 만듭니다. 포함 내용은 다음과 같습니다.
* 클래스와 결함
* review scope
* 프레임별 라벨 (부분 라벨 상태와 가시성 정보 그대로)

CVAT 변환기는 구현하지 않았습니다(LIMITATIONS 참조).

## 합성 데이터 세션 파일

* `angles.csv`
* `session_meta.json`
* `ground_truth.json`: 모든 결함. 라벨하지 않은 결함 포함
* `labels.json`: 부분 라벨
