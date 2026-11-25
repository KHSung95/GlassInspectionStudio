# 사용자 매뉴얼

## 1. 프로젝트 만들기와 이미지 가져오기

1. **File → New project…**에서 빈 폴더를 고릅니다. 기존 프로젝트를 덮어쓰지 않습니다.
2. **File → Import image folder… (Ctrl+I)**에서 폴더와 레이아웃을 고릅니다.
   * `bottle/session`: `<root>/<병코드>/<세션>/<프레임>`
   * `bottle`: `<root>/<병코드>/<프레임>` (세션 코드는 `S001`)
   * `flat`: `<root>/<프레임>` (병 코드를 직접 입력)
3. 각도는 아래 순서로 정해집니다.
   1. `angles.csv` (`filename,angle[,frame_index]`)
   2. 파일명 정규식 (`_a12.5`, `angle_30`, `15deg` 등)
   3. `순서 × angle step`
   * 세션 폴더에 `session_meta.json`이 있으면 카메라, 비트 심도, 노출, 게인, 조명, 렌즈, 배율, mm/px를 함께 읽습니다.
4. 원본 파일은 **경로와 SHA-256으로만 참조**합니다. 복사하거나 수정하지 않습니다.
   * 해시가 같은 파일은 중복으로 보고 건너뜁니다.
   * **File → Find duplicate frames**에서 중복 목록을 볼 수 있습니다.
5. 지원 포맷은 PNG, BMP, TIFF, JPG, PGM입니다.
   * 8bit와 16bit 컨테이너를 지원합니다.
   * 10/12bit 센서 값은 원래 값 그대로 유지합니다.
   * 유효 비트 심도는 세션 메타데이터(`bit_depth`)로 지정합니다.

Dataset Manager 사용법:
* 검색과 필터: 병 코드, 로트, 메모, 품종, 판정 상태로 찾습니다.
* 썸네일: 디스크 캐시에서 필요할 때마다 불러옵니다.
* Properties 도크: 병과 세션의 메타데이터를 편집합니다.
* **File → Compare capture sessions**: 같은 병을 반복 촬영한 세션들을 비교합니다.

## 2. 이미지 뷰어

| 동작 | 방법 |
|---|---|
| 확대/축소 | 마우스 휠 (커서 위치 기준) |
| Pan | 가운데 버튼 드래그, 또는 Pan 도구 (H) |
| 화면 맞춤 / 1:1 원본 픽셀 | `F` / `Ctrl+1` |
| 픽셀 그리드 / 원본 값 표시 | 12배 이상 확대 시 그리드, 40배 이상 확대 시 각 픽셀의 원본 값 |
| 좌표·밝기 | Display 도크에 x, y, 픽셀 인덱스, 원본 값, 점수맵 값 표시 |
| ROI 통계 | Measure 도구 (M)로 사각형을 그리면 mean/std/min/max/median 표시 |
| 표시 변환 | Full range / 0.5–99.5% stretch / Window·Level, Gamma, CLAHE, Invert, Histogram |
| 비교 | 툴바 *Compare*: 이전·다음 프레임, 원본 full range, 점수맵. 확대 좌표가 동기화됨 |

표시 변환(`DisplayTransform`)은 화면에만 적용됩니다. 학습과 추론 입력(`InputNormalization`)과는 완전히 분리되어 있습니다.

## 3. 회전 시퀀스 탐색 (Rotation Timeline)

* `←` / `→`: 이전·다음 프레임. 0°와 360°는 순환으로 이어집니다.
* 각도 입력칸: 원형 거리로 가장 가까운 프레임으로 이동합니다. 예를 들어 359°를 입력하면 0°로 갑니다.
* `Space`: 연속 재생. 재생 속도(fps)는 조절할 수 있습니다.
* 그래프: 선택한 Run의 각도별 최대 후보 점수를 보여줍니다. 그래프를 클릭하면 해당 각도로 이동합니다.
* 프레임을 바꿔도 확대 위치는 유지됩니다. 같은 좌표를 여러 관찰각에서 비교할 수 있습니다.

## 4. Annotation Studio

### 라벨 상태

| 상태 | 의미 | 학습 시 처리 |
|---|---|---|
| Positive | 확인된 결함 | 해당 클래스 채널의 정답 |
| Negative | 사람이 확인한 정상·허용 영역 | 그 영역은 모든 결함 채널에서 0 (Hard negative) |
| Unknown | 판정 불가 | Loss에서 제외 |
| Ignore | 학습과 평가 모두에서 제외 | Loss와 평가 모두에서 제외 |

* **라벨이 없는 모든 픽셀은 Unknown입니다.**
  * NG 병이라고 해서 모든 프레임이 NG가 되지 않습니다.
  * Positive가 없는 프레임이라고 해서 정상이 되지도 않습니다.
* **Frame review → "Exhaustively reviewed for"**에서 클래스를 체크하면, 그 프레임에서 해당 클래스 채널에 한해서만 라벨이 없는 영역을 Negative로 취급합니다.
* Dust와 Scratch는 관측 외관 클래스입니다. 최종 판정은 *Set spec verdict*(`undecided`/`ng`/`ok`)로 따로 관리합니다.
  * `undecided`인 동안 그 영역은 결함 채널에서 Unknown으로 처리됩니다.
* 기포의 열림·닫힘처럼 영상만으로 정할 수 없는 정보는 Defect의 `open_state=undetermined`로 둘 수 있습니다.

### 도구와 단축키

| 키 | 도구 |
|---|---|
| V | 선택 |
| H | Pan |
| P | Point (극미세 흑점은 중심을 클릭) |
| B | Bounding box |
| G | Polygon (더블클릭 또는 Enter로 완료) |
| R | Brush mask (반경 설정 가능) |
| O | Normal ROI (Negative) |
| I | Ignore ROI |
| M | Measure |

* 클래스 단축키: `1`–`9`, `D`(Dust), `S`(Scratch), `N`(Verified Normal), `U`(Unknown). 클래스 추가·수정은 *Add / edit class…*에서 합니다.
* `Ctrl+Z` / `Ctrl+Y`: Undo / Redo
* `Del`: 삭제 (soft delete, 이력 보존)
* `Ctrl+F`: 라벨 검색
* `C` / `L`: 후보 / 라벨 표시 토글
* 모든 변경은 즉시 SQLite에 저장됩니다(자동 저장). 변경 이력은 `annotation_history`에 남습니다.
* 겹치는 라벨, 특히 상태가 서로 충돌하는 라벨은 Frame review 영역에 표시됩니다.

### 결함 Tracking

1. 라벨을 고르고 **New defect ID**를 누르면 물리적 결함 ID가 부여됩니다. 다른 프레임의 라벨은 **Link to defect…**로 연결합니다.
2. **Set visibility**로 Visible / Weak / Not visible / Uncertain을 지정합니다. **Keyframe**으로 대표 프레임을 지정합니다.
3. 다른 프레임으로 전파하는 방법은 세 가지입니다.
   * **Copy → next N**: 다음 N개 프레임에 복사
   * **Interpolate 2 sel.**: 두 Keyframe 사이를 선형 보간
   * **Track (template)**: 템플릿 매칭으로 다음 프레임 추적
4. 전파된 라벨은 모두 `confirmed=0`(점선 표시)로 저장되며 GT에서 제외됩니다. 사용자가 **Confirm**해야 GT가 됩니다. 굴절 때문에 움직임이 비선형이고 가시성도 바뀌기 때문입니다.

### Point → Heatmap

* 학습 heatmap의 Gaussian sigma는 Training Lab의 *point heatmap sigma*로 정합니다.
* 이것은 학습 타깃의 모양일 뿐이며, 결함의 물리적 크기가 아닙니다.
* 실제 크기는 평가에서 `attributes.size_px`나 Angular analysis의 apparent size로 다룹니다.

## 5. Classical Vision Lab

* 흑점 검출: Black-hat, DoG, Multi-scale LoG, Local background subtraction, Local contrast(z-score), Adaptive threshold + CCA
* 명암·투명도 이상: Multi-scale low-frequency deviation, Local intensity deviation, Gradient/texture
* 기포: blob + contour + ellipse/circularity + bright/dark ring 분석

사용 방법:
* 점수는 대부분 **SNR**(응답 ÷ 강건한 노이즈 σ)이라 비트 심도와 무관합니다.
* *Live preview*를 켜면 파라미터를 바꿀 때마다 현재 프레임에 다시 적용합니다. 이 결과는 저장되지 않습니다.
* 실행 단위:
  * *Run on frame*: Run으로 저장
  * *Run on ROI*: 마지막 Measure 사각형 영역만
  * *Run batch*: 현재 세션, 현재 병, 또는 전체
* *Save/Load preset*으로 파라미터를 저장하고 불러옵니다.
* 모든 저장 Run에는 알고리즘, 파라미터, 전처리, 처리 시간, 점수 의미가 기록됩니다.
* 결과는 **후보**이며 GT가 아닙니다.

### 5.1 Sweep & recommend (라벨 기반 파라미터 추천)

Classical Vision Lab의 **Sweep & recommend…** 버튼을 누르면 확정 라벨에 맞춰 파라미터를 탐색합니다. CLI는 `gistudio.cli sweep`입니다.

1. 장면에 뚜렷한 흑점이 있으면 Point 도구(P)로 몇 개 찍습니다.
2. 눈금, 숫자, 경계처럼 오검출되기 쉬운 정상 구조물은 Normal ROI(O)로 표시합니다.
   * 이 영역이 있어야 오검출(FP)을 셀 수 있습니다.
3. 탐색 범위(JSON)를 정합니다. 알고리즘마다 기본값이 있습니다.
   * Threshold는 탐색 범위에 넣지 않아도 됩니다. 각 조합을 한 번만 실행한 뒤 후보 점수에서 자동으로 sweep합니다.
4. 프레임마다 라벨 주변 crop(+margin)만 처리합니다.
   * 프레임 전체의 노이즈 σ를 넘겨주므로, 추천 threshold를 전체 프레임에 그대로 쓸 수 있습니다.
5. 추천 프리셋은 목적별로 나옵니다.

| 프리셋 | 기준 |
|---|---|
| balanced | F1 최대 |
| high_recall | recall ≥ 목표를 만족하면서 FP·unverified 최소 |
| low_fp | 검증된 FP/frame ≤ 예산을 만족하면서 recall 최대 |
| zero_verified_fp | 검증된 FP 0, recall 최대 |

   * low_fp와 zero_verified_fp는 recall이 0.3 미만이면 추천하지 않고 경고를 띄웁니다.
   * 비교 기준으로 기본값(defaults)의 결과도 함께 표시합니다.
6. 병이 2개 이상이면 *Bottle hold-out*으로 일부 병을 검증용으로 남겨, 과적합 여부를 볼 수 있습니다.
7. 결과 활용:
   * *Apply selected to lab*: 선택한 프리셋을 Lab 폼에 적용
   * *Save all as presets*: `presets/`에 저장. 지표, GT 수, 생성 시각이 함께 기록됨

추천값은 부분 라벨로 만든 권고입니다. 검증을 마친 양산용 설정이 아닙니다.

### 5.2 Acquisition (카메라 촬영)

* GenICam 카메라를 Harvesters와 Spinnaker GenTL producer로 연결합니다.
  * 검증한 장비: JAI GOX-12405M-5GE, GigE
* **SpinView가 카메라를 열고 있으면 연결할 수 없습니다.** 카메라 하나는 한 앱만 제어할 수 있습니다.
* *Connect* 후의 동작:
  * 현재 카메라 설정을 그대로 읽기만 합니다.
  * Pixel format, Exposure, Gain은 **Apply를 누를 때만** 카메라에 씁니다.
* *Live view*는 저장되지 않습니다. 촬영 전 품질 확인용으로 다음 값을 함께 보여줍니다.
  * 초점 지표 (중앙 영역의 Laplacian 분산)
  * 포화 %, 암부 %, 평균, 최대값
* 촬영 순서:
  1. Bottle code를 입력하고 *New session*을 누릅니다.
  2. 병을 손으로 회전합니다.
  3. *Capture (F8)*를 누릅니다. 누를 때마다 각도가 step만큼 증가합니다.
* 저장 위치는 `<project>/acquired/<bottle>/<session>/`입니다.
  * 저장되는 파일: 무손실 TIFF, `angles.csv`, `session_meta.json`, `camera_features.json`
  * 프레임은 저장되는 즉시 프로젝트 DB에 등록됩니다.
  * 기존 세션 폴더는 덮어쓰지 않습니다.
* *Average frames*: 정지 장면의 노이즈를 줄이기 위해 여러 장을 평균합니다.
* CLI 명령:
  * 카메라 목록: `gistudio.cli cameras`
  * 촬영: `gistudio.cli capture <project> --bottle NG-001 --n 12 --step 30 --interactive`
* **웹 카메라 콘솔:** 브라우저에서 실시간 영상을 보고 바로 촬영할 수 있습니다. 이 PC(127.0.0.1)에서만 접속됩니다.
  ```powershell
  .\.venv\Scripts\python.exe -m gistudio.cli console C:\dev\gis_pilot --port 8765
  ```
  * 실행한 뒤 `http://127.0.0.1:8765`를 엽니다.
  * 실시간 영상: MJPEG 미리보기입니다. 화면 표시용 자동 대비는 저장되는 영상에 영향을 주지 않습니다.
  * 원본 확대: 영상을 클릭하면 그 위치를 원본 해상도(1:1)로 확대해 보여줍니다. 국소 초점 지표와 min/mean/max도 함께 표시합니다.
  * 촬영: 세션을 만들고 Space 또는 *촬영* 버튼을 누릅니다. 원본 TIFF로 저장되고 즉시 프로젝트에 등록됩니다.
  * 카메라 설정: Apply를 누르고 확인 창에서 승인할 때만 카메라에 기록됩니다.
  * 콘솔이 카메라를 점유하는 동안에는 데스크톱 GUI, CLI, SpinView가 카메라에 연결할 수 없습니다. 콘솔을 종료하면 카메라가 해제됩니다.
* 파일럿 리포트(고전 CV 기준):
  ```powershell
  .\.venv\Scripts\python.exe tools\pilot_inference.py <project> --bottle NG-001 --sweep
  ```
  * 결과는 `exports/pilot_*`에 리포트(md/json)와 오버레이 PNG로 저장됩니다.
  * 모델은 `--unet <ckpt>`, `--winclip`을 줄 때만 실행됩니다.

## 6. Zero-shot Inference Lab

* **WinCLIP (Anomalib 2.7.1, zero-shot k=0)**와 **OpenCLIP patch-prompt** 모드를 지원합니다.
* 원본 해상도 타일(tile, overlap)을 사용하며, 결과는 원본 좌표로 복원합니다.
  * Resize 비율, 모델 feature stride, 원본 기준 유효 stride를 경고로 표시합니다.
  * 1–3px 흑점은 CLIP 계열에서 놓칠 가능성이 높습니다.
* Prompt는 편집하고 저장·불러올 수 있습니다(JSON `{normal:[...], defect:[...]}`). 결과의 `model_info.mean_prompt_prob`로 Prompt별 결과를 비교할 수 있습니다.
* 점수는 보정된 확률이 아닙니다. 합성 데모에서 WinCLIP 점수는 0.32–0.40 범위에 몰려 있었습니다. 임계값은 데이터마다 다시 정해야 합니다.

## 7. Training Lab

1. **Bottle-level split**을 만듭니다(physical sample ID 기준).
   * 같은 물리 병의 반복 촬영과 다른 각도는 반드시 같은 subset에 들어갑니다.
   * *hold-out*에 품종을 입력하면 그 품종 전체를 test로 보내는 일반화 시험이 됩니다.
2. 모델을 고릅니다: U-Net(HRUNet) 또는 Multi-scale Patch CNN.
3. 하이퍼파라미터를 정합니다: epoch, batch, LR, tile, samples/epoch, base channels, 채널별 loss, sigma, device, AMP, seed.
4. **Start training**을 누르면 별도 프로세스에서 학습이 진행되고 GUI는 멈추지 않습니다.
   * 진행률과 loss curve가 표시됩니다.
   * **Stop**을 누르면 현재 step을 마친 뒤 정지하고 checkpoint는 유지됩니다.
   * 중단한 학습은 `last.pt` 경로를 *resume*에 넣으면 재개됩니다.
5. 산출물은 `runs/<run>/` 아래에 생깁니다.
   * `best.pt`, `last.pt`
   * `config.json`
   * `metrics.jsonl`
   * `tb/` (TensorBoard 기록: `.venv\Scripts\tensorboard --logdir <run>\tb`)

## 8. 추론, 평가, 리뷰

* **Model Inference** 도크: 학습된 checkpoint로 U-Net 추론이나 Classical + Patch CNN cascade를 실행합니다.
* **Evaluation Lab → Model comparison**: 같은 프레임 집합(split subset)에서 여러 Run을 비교합니다.
  * 비교 지표: Recall, 검증된 FP, unverified, FP/frame, FROC, 위치 오차, 결함·프레임·병 단위 결과, 처리 시간
  * Unknown 영역의 검출은 FP가 아니라 **unverified**로 따로 셉니다.
  * 정상병이 없으면 오배출률은 "NOT MEASURABLE"로 표시됩니다.
* **Angular visibility**: 결함별로 다음을 계산합니다.
  * 각도별 점수, 국소대비, 크기
  * Best view와 가시 구간 (0/360° 원형 처리)
  * 촬영 간격 재샘플링(1–90°) × 시작각에 따른 평균·최악 미검률, 필요한 fps
* **Multi-view**: 병 단위로 집계하고 NG 근거 이미지를 저장합니다.
  * 집계는 최대 유효 점수 기준이며 다수결은 쓰지 않습니다.
  * 저장 위치: `exports/evidence_run<N>/`
* **Active Learning Review**: 큐 전략은 다음 중에서 고릅니다.
  * high_score, uncertain
  * label_disagreement, on_normal_structure, new_unlabelled
  * model_disagreement, random
* 리뷰 결정과 결과:
  * **A** Accept → Positive GT (`source=ai_accepted`)
  * **X** Reject → Hard negative ROI
  * **Shift+U** → Unknown
  * *Apply to group*을 켜면 multi-view 그룹 전체에 한 번에 적용됩니다.
