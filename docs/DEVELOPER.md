# 개발자 매뉴얼

## 아키텍처

```
gistudio/
  core/            project.py (Project/DB facade), db.py (schema), image_io.py (무손실 IO, LRU cache),
                   preprocessing.py (DisplayTransform ↔ InputNormalization 분리), geometry.py (좌표/원형각/래스터화),
                   tiling.py (타일/Stitcher), annotations.py (AnnotationStore: undo/redo/history/defect/propagate),
                   label_targets.py (부분 라벨 → target+valid), splits.py (병 단위 split), dataset_import.py,
                   thumbnails.py, tracking.py, device.py, export.py
  algorithms/      base.py (InspectionAlgorithm, Candidate, InspectionResult, Registry),
                   classical/ (ops.py, algorithms.py), zero_shot/ (common.py, plugins.py), supervised/ (models.py, plugins.py)
  training/        losses.py (masked BCE/Dice/Focal/heatmap focal/MSE), augmentations.py, datasets.py, trainers.py
  inference/       engine.py (배치 추론·저장), multiview.py (병 단위 집계), candidate_management.py (AL 큐)
  evaluation/      metrics.py (부분 라벨 평가, FROC), angular.py (각도 가시성·재샘플링)
  acquisition/     interface.py (CameraProvider/RotationStageProvider/LightController/CaptureCoordinator), mock.py
  app/             PySide6 GUI (state.py, viewer.py, panels_*.py, main_window.py)
  synthetic.py     합성 회전 시퀀스 생성기 (기능 검증용)
  cli.py           CLI
```

원칙:
* GUI는 core, algorithms, training 등을 호출하기만 합니다.
* 학습은 `QProcess`로 `python -m gistudio.cli train-unet ...`을 실행하는 **별도 프로세스**입니다.
* 프레임 단위 추론과 평가는 `QThread`(Worker)에서 실행합니다.
* SQLite 연결은 스레드마다 따로 엽니다(`Database.conn`이 thread-local).

## 새 알고리즘 추가

1. `InspectionAlgorithm`을 상속하고 `algorithm_id`, `family`, `score_semantics`, `params_spec`, `_run()`을 구현합니다.
2. `@register`를 붙입니다. 등록 방법은 세 가지입니다.
   * `gistudio/algorithms/...` 안에 두고 `base._load_builtin()`에 import를 추가
   * 저장소 루트의 `plugins/*.py`에 둠 (자동 로드, 실패해도 앱은 계속 실행)
   * entry point 그룹 `gistudio.algorithms`로 등록
3. `_run(image, meta)`의 규칙:
   * 입력: 원본 dtype의 단일 채널 이미지, `meta`(`bit_depth`, `frame_id`, `angle_deg`, `progress`)
   * 출력: `InspectionResult(candidates, score_maps, preprocess, model_info, warnings)`
   * 모든 좌표는 원본 이미지 좌표여야 합니다.
4. GUI 파라미터 폼은 `params_spec`에서 자동으로 만들어집니다. GUI 코드는 수정하지 않아도 됩니다.

예시: [examples/example_plugin.py](examples/example_plugin.py)

## 새 학습 모델 추가

* 모델 클래스는 `algorithms/supervised/models.py`에 둡니다. checkpoint `meta.arch`로 복원합니다(`plugins.load_checkpoint`).
* Trainer는 `training/trainers.py`의 패턴을 따릅니다.
  * 진행 상황: `emit()` → `@@PROGRESS` JSON
  * `last.pt`와 `best.pt` 저장
  * `STOP` 파일로 정지
  * checkpoint `meta`에 정규화와 타깃 정책 저장
* 손실 함수는 반드시 `valid` 마스크를 받아 Unknown을 제외해야 합니다(`training/losses.py`).

## 정상 기반 모델(PatchCore / EfficientAD) 연동 지침

* 학습 데이터는 `label_state='negative'`(사람이 검증한 정상) 영역과, review scope로 전수 검토된 프레임에서만 뽑습니다.
* **NG 병의 미확인 영역을 정상으로 쓰지 않습니다.**
* Anomalib `Patchcore` / `EfficientAd` Lightning 모델을 `anomaly_detection` family 플러그인으로 감싸고, `tiled_map`으로 원본 해상도를 유지하는 방식을 권장합니다.
* 검증된 정상 ROI만으로 학습한 결과는 "정상병 전체를 대표하지 못함" 경고와 함께 기록합니다.

## 하드웨어 연동

* `acquisition/interface.py`의 추상 클래스를 구현하는 SDK 어댑터를 별도 모듈로 만듭니다.
  * 예: Harvesters/GenICam, Spinnaker PySpin, 모터 컨트롤러
* SDK import는 그 모듈 안에서만 합니다. SDK가 없어도 앱은 동작합니다.
* `CaptureCoordinator.capture_sequence()`는 비어 있지 않은 폴더에 쓰지 않습니다(데이터 보호).

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not slow"
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m slow
```

```powershell
.\.venv\Scripts\python.exe tools\gui_smoke.py C:\dev\gis_demo_copy C:\dev\gui_shots
```

* 첫 번째 명령: 빠른 테스트
* 두 번째 명령: WinCLIP/OpenCLIP 실제 추론 (가중치 필요)
* 세 번째 명령: 헤드리스 GUI 스모크 (프로젝트 사본 사용 권장)

## 의존성 고정 메모

* `open-clip-torch`는 Anomalib 2.7.1이 선언한 범위(`>=2.23.0,<2.26.1`)에 맞춰 **2.24.0**으로 고정했습니다.
  * open_clip 4.x에서는 `WinClipModel.encode_image`가 `ValueError: too many values to unpack`로 실패하는 것을 확인했습니다.
* `opencv-python-headless`를 사용합니다. PySide6와 Qt 플러그인이 충돌하지 않게 하기 위해서입니다.
