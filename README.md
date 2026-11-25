# Glass Inspection Studio

투명 의약품 유리병의 미세 결함 검사를 위한 **로컬 머신비전 연구개발 플랫폼**입니다.
이미지 관리, 부분 라벨링, 고전 영상처리, Zero-shot 추론, 지도학습, 모델 비교, Active Learning을 하나의 데스크톱 애플리케이션(PySide6)과 CLI에서 수행합니다.

> 모든 이미지와 검사 데이터는 로컬에만 저장됩니다. 클라우드 API로 이미지를 보내지 않습니다.
> Zero-shot·모델 점수는 보정된 불량 확률이 아니라 검증되지 않은 **후보**입니다.
> 합성 데이터에서 나온 수치는 실제 광학계의 검출 성능을 대표하지 않습니다.

## 빠른 시작

```powershell
cd C:\dev\GlassInspectionStudio
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -DownloadModels -RunTests
```

```powershell
.\.venv\Scripts\python.exe -m gistudio.cli demo C:\dev\gis_demo
```

```powershell
.\launch.ps1 C:\dev\gis_demo
```

* `setup.ps1`은 프로젝트 전용 `.venv`만 만들고, 시스템 전역 Python은 건드리지 않습니다.
* `nvidia-smi`로 NVIDIA 드라이버를 감지해 CUDA 빌드(cu118/cu126/cu128)를 설치합니다. 드라이버가 없으면 CPU 빌드를 설치합니다.
* 의존성 버전은 `requirements.lock.txt`에 고정되어 있습니다. PyTorch는 `setup.ps1`의 `$TorchVersion`으로 고정합니다.
* 프로젝트 경로는 짧게 유지하세요. Windows 경로 길이 제한 때문에 PyTorch DLL 로드가 WinError 206으로 실패할 수 있습니다.

## 구성

| 모듈 | GUI 도크 | 핵심 코드 |
|---|---|---|
| A. Dataset Manager | Dataset Manager, Properties | `gistudio/core/project.py`, `dataset_import.py`, `thumbnails.py` |
| B. Annotation Studio | Annotation Studio, Display/Histogram, Rotation Timeline | `core/annotations.py`, `core/label_targets.py`, `app/viewer.py` |
| C. Classical Vision Lab | Classical Vision Lab | `algorithms/classical/` |
| D. Zero-shot Inference Lab | Zero-shot Inference Lab | `algorithms/zero_shot/` (WinCLIP, OpenCLIP prompts) |
| E. Training Lab | Training Lab | `training/` (HRUNet, Patch CNN, masked loss) |
| F. Evaluation Lab | Evaluation Lab (비교, Angular, Multi-view) | `evaluation/`, `inference/multiview.py` |
| G. Active Learning Review | Active Learning Review | `inference/candidate_management.py` |
| 하드웨어 추상화 | – | `acquisition/` (Mock, File provider) |

모든 내부 로직은 GUI와 독립적으로 동작합니다. 무거운 작업은 모두 CLI로도 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m gistudio.cli --help
```

## 문서

* [사용자 매뉴얼](docs/USER_MANUAL.md): 워크플로, 단축키, 라벨링 규칙
* [개발자 매뉴얼](docs/DEVELOPER.md): 아키텍처, 알고리즘·모델 추가 방법
* [데이터 포맷 명세](docs/DATA_FORMAT.md)
* [모델 검증 기록](docs/MODELS.md): 출처, 버전, 라이선스, 통합 여부
* [검증 결과](docs/VALIDATION.md): 실제 실행한 테스트와 결과
* [알려진 한계 및 미구현 기능](docs/LIMITATIONS.md)
