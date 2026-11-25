# 모델 검증 기록

모든 가중치의 출처·해시·라이선스는 `tools/download_models.py`가 `models/model_registry.json`에 기록합니다.
가중치는 `models/clip_cache/`에만 저장됩니다. open_clip이 기본으로 쓰는 `~/.cache/clip` 경로는 사용하지 않도록 리다이렉트했습니다.

| 모델 | 상태 | 구현체 / 출처 | 비고 |
|---|---|---|---|
| WinCLIP (zero-shot, k=0) | **통합·실행 검증** | `anomalib.models.image.winclip.torch_model.WinClipModel` (Anomalib 2.7.1, Apache-2.0) | backbone ViT-B-16-plus-240 / laion400m_e31. 입력 240px, patch stride 16 |
| OpenCLIP patch-prompt | **통합·실행 검증** | open_clip_torch 2.24.0 (MIT) | 위와 같은 가중치 공유. 타일당 1개 임베딩 |
| AnomalyCLIP | **미통합** | 공식 저장소(zqhang/AnomalyCLIP) | pip 패키지가 없고 checkpoint가 저자 저장소/외부 드라이브로 배포됨. 이 환경에서 공식 checkpoint의 무결성과 라이선스를 검증하지 못해 placeholder(`zeroshot.anomalyclip`)만 등록. 호출하면 명확한 오류를 냄 |
| DINO 계열 / SAM 계열 | 미통합 (플러그인 구조만) | – | SAM은 라벨링 보조로만 사용해야 함. `InspectionAlgorithm` 플러그인으로 추가 |
| PatchCore / EfficientAD | 미통합 (지침만) | Anomalib에 포함 | 검증된 정상 데이터가 필요. DEVELOPER.md 참조 |
| HRUNet / Multi-scale Patch CNN | 자체 구현 | `gistudio/algorithms/supervised/models.py` | 처음부터 학습 (사전학습 가중치 없음) |

## WinCLIP 가중치

* URL: `https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt`
* 출처: open_clip 공식 GitHub release
* SHA-256: `8fb26589b9a8bab2e7a683d280e48df89b8e742f3a82132707282620b36facba` (파일명에 들어 있는 prefix와 일치함을 확인)
* 라이선스:
  * open_clip 코드: MIT
  * 가중치: LAION-400M으로 학습됨. 상업적으로 배포하기 전에 LAION 데이터셋 조건을 검토해야 합니다.

## 해상도 관련 주의

* WinCLIP: tile 240px을 그대로 쓰면 resize 비율은 1.0, 원본 기준 feature stride는 16px입니다.
  * tile을 480px로 키우면 0.5배로 축소되어 유효 stride가 32px이 됩니다.
  * 실행할 때마다 이 값을 `model_info`와 경고로 남깁니다.
* 1–3px 흑점은 16px 패치 안에서 신호 비중이 매우 작습니다. CLIP 계열로는 미검 가능성이 높다고 전제합니다.
* 합성 데모에서 WinCLIP 점수 분포: min 0.32, p50 0.36, max 0.40 (프레임 37, 38). 점수 범위가 좁고 보정되어 있지 않습니다.
