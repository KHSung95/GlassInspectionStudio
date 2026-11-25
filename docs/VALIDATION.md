# 검증 결과 (2026-11-24, 실제 실행 기록)

## 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Home (10.0.26300) |
| CPU / RAM | Intel Core Ultra 5 226V / 15.5 GB |
| GPU | Intel Arc 130V. NVIDIA GPU 없음, `nvidia-smi` 없음 → **CPU 빌드** |
| Python | 3.11.9 (프로젝트 `.venv`) |
| 주요 패키지 | torch 2.15.0+cpu, torchvision 0.29.2, PySide6 6.12.0, opencv-python-headless 5.1.0.95, numpy 2.4.6, tifffile 2026.3.3, anomalib 2.7.1, open-clip-torch 2.24.0 |
| 전체 고정 목록 | `requirements.lock.txt` |

## 자동 테스트: `python -m pytest -q` → **41 passed** (slow 테스트 포함, 55초)

| 요구 테스트 | 테스트 |
|---|---|
| 16bit 원본 데이터 보존 | `test_image_io.py::test_16bit_tiff_and_png_roundtrip_exact`, `test_import_never_modifies_originals` (SHA·mtime 불변, 캐시 read-only, 표시 변환 비파괴) |
| 이미지 좌표 및 확대 좌표 변환 | `test_view_transform_roundtrip_and_zoom_anchor`, `test_pixel_index_convention`, `test_qt_viewer_mapping_matches_image_pixels` (실제 QGraphicsView, 0.5×–33×) |
| 타일링과 원본 좌표 복원 | `test_tiles_cover_image_and_restore_exactly` (5개 크기 × mean/window/max), `test_tile_local_coordinates_map_to_original`, `test_pad_to_keeps_original_region` |
| Unknown Loss Mask | `test_unlabelled_pixels_are_unknown_not_normal`, `test_negative_roi_scope_ignore_unknown_rules`, `test_masked_loss_ignores_unknown_region` (5개 loss 모두 Unknown 영역 gradient = 0) |
| 유효 라벨이 없는 배치 | `test_batch_without_valid_pixels_is_safe` |
| Bottle ID 기반 Split 누출 방지 | `test_split_by_physical_sample_no_leakage` (같은 physical sample → 같은 subset, 고의로 오염시키면 감지), `test_holdout_product_type` |
| 라벨 저장 및 복구 | `test_labels_persist_across_reopen_with_history_and_undo` (point/polygon/mask, undo, history, 재오픈) |
| AI 후보와 Ground Truth 분리 | `test_ai_candidates_are_not_ground_truth`, `test_propagated_labels_unconfirmed_until_reviewed` |
| 회전각 0/360도 경계 | `test_circular_angles`, `test_visible_interval_crossing_zero_is_merged`, `test_appstate_navigation_wraps`, `test_sampling_experiment_start_offsets` |
| 합성 데이터 학습·추론 Smoke | `test_unet_train_infer_eval_review_cycle` (학습 → resume → 추론 → 평가 → accept/reject), `test_patch_cnn_train_and_cascade`, `test_classical_detects_synthetic_spots`, `test_all_classical_algorithms_run`, `test_multiview_and_angular_analysis` |
| 재현성 | `test_training_is_reproducible_with_seed` (같은 seed → 같은 loss) |
| Zero-shot 실제 실행 | `test_winclip_tiled_inference_runs`, `test_clip_patch_prompts_run` (로컬 가중치, `HF_HUB_OFFLINE=1`) |
| 평가 규칙 | `test_detections_in_unknown_area_are_not_false_positives` (TP 1, verified FP 1, unverified 1, 오배출률 NOT MEASURABLE) |

## GUI 검증: `tools/gui_smoke.py` (offscreen Qt, 데모 프로젝트 사본)

다음 항목을 확인했습니다.
* 프로젝트 열기와 원형 프레임 탐색
* 각도 입력(365° → 0°)
* Point와 Normal ROI 라벨 저장, Undo/Redo
* Classical live preview (후보 31개)
* 이전 프레임 비교 뷰
* 60배 확대 시 픽셀 그리드와 원본 값 표시
* **GUI에서 시작한 U-Net 학습** (별도 QProcess, exit code 0, `best.pt` 생성)
* Evaluation Lab 비교 표 생성

화면 캡처를 보고 도크 레이아웃과 썸네일 버그를 찾아 고쳤습니다. 실제 모니터와 마우스로 하는 수동 조작은 이 세션에서 하지 않았습니다.

## 합성 데모 (CLI `demo`, 6병 × 36각 × 640×512, test split = 2병·72프레임)

**기능 검증용 수치이며 실제 성능이 아닙니다.**

| Run | Recall (op. thr) | 검증된 FP | Unverified | 위치 오차 평균 | 물리 결함 recall (any view) | ms/frame (CPU) |
|---|---|---|---|---|---|---|
| classical.log_blob (SNR ≥ 6) | 0.864 | 108 (주로 정상 눈금 ROI) | 2692 | 1.64 px | 1.0 | 82 |
| supervised.unet 25 epoch (≥ 0.3) | 0.983 | 1 | 9696 | 1.49 px | 1.0 | 662 |
| zeroshot.winclip (4프레임) | 0 후보 (≥ 0.5) | – | – | – | – | ≈47,000 |

해석과 주의점:
* U-Net의 unverified 후보가 많습니다. 대부분 라벨이 없는(Unknown) 영역의 area 채널 반응입니다. 리뷰 큐로 확인해야 하는 대상이며, FP로 단정할 수 없습니다.
* WinCLIP 점수 범위는 0.32–0.40으로 좁아서, 기본 임계값 0.5에서는 후보가 나오지 않았습니다.
* 짧은 학습(6 epoch)에서는 heatmap focal loss의 양성 픽셀이 0이 되는 버그가 있어 recall이 0이었습니다. 수정한 뒤 위 결과를 얻었습니다(아래 수정 이력 1번).

## 개발 중 발견하고 수정한 문제

1. Gaussian heatmap 피크가 1.0에 도달하지 않음 → heatmap focal loss의 양성 픽셀이 0이 되어 학습 실패. 점이 속한 픽셀을 1.0으로 고정해 수정.
2. GroupNorm 그룹 수가 채널 수의 약수가 아닐 때 생기는 오류 수정 (base_channels=12 등).
3. Patch CNN에 검증 데이터가 없으면 `best.pt`가 저장되지 않던 문제 (best 초기값 오류) 수정.
4. Anomalib 2.7.1 WinCLIP과 open_clip 4.x가 호환되지 않음 (`encode_image` 반환값 개수) → Anomalib이 선언한 범위의 open_clip 2.24.0으로 고정.
5. open_clip 2.24가 가중치를 `~/.cache/clip`에 저장하던 문제 → `models/clip_cache`로 리다이렉트하고 SHA-256 검증.
6. GUI: 저장된 도크 상태가 중앙 뷰어를 가리던 문제, 썸네일이 잘리던 문제, 고배율 픽셀 값이 표시되지 않던 문제 수정.
7. Windows 긴 경로(WinError 206)로 torch DLL 로드 실패 → 프로젝트를 `C:\dev\GlassInspectionStudio`로 옮기고, `setup.ps1`에 경고를 추가.

## 최소 완료 조건 대응

| # | 조건 | 결과 |
|---|---|---|
| 1 | Import | ✅ CLI/GUI, 레이아웃 3종, CSV/정규식 각도, 중복 탐지 |
| 2 | 고해상도 원본 열기와 회전각 탐색 | ✅ (GUI smoke, 탐색 테스트) |
| 3 | 실제 이미지 좌표로 라벨 저장 | ✅ |
| 4 | Unknown을 자동으로 정상 처리하지 않음 | ✅ |
| 5 | 고전 영상처리로 후보 생성 | ✅ 10종 |
| 6 | 실제 Zero-shot 모델 실행 | ✅ WinCLIP(Anomalib) + OpenCLIP |
| 7 | Masked Loss 기반 U-Net 학습 | ✅ |
| 8 | 학습한 모델로 새 이미지 추론 | ✅ (test split = 학습에 쓰지 않은 병) |
| 9 | 추론 후보를 검토해 학습 데이터에 추가 | ✅ accept → GT, reject → hard negative |
| 10 | 병 단위 split | ✅ |
| 11 | 각도별 검출 분석 | ✅ profile, 가시 구간, 재샘플링 미검률 |
| 12 | 실험 설정·결과 저장과 재현 | ✅ run/config/metrics 기록, seed 재현성 테스트 |
| 13 | 재실행 후 복원 | ✅ 재오픈 테스트, last_project 자동 열기 |
| 14 | 문서만으로 workflow 재현 | ✅ README + USER_MANUAL. 단, 깨끗한 PC에서 setup.ps1을 처음부터 실행해 보지는 않았음 (기존 venv에 재실행만 검증) |
