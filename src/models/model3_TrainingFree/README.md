# 중고 상품 사진 AI 검사

이 폴더의 파일:

- `detect.py` — 기준 저장, 판정, AUROC 실행
- `requirements.txt`
- `detector/haar.py` — WaRPAD용 Haar 고주파
- `detector/images.py` — 이미지 열기·중앙 크롭
- `detector/rdd.py` — AeroBlade 재구성 오차와 RDD 디바이어싱
- `detector/warpad.py` — DINOv2 고주파 민감도
- `detector/scores.py` — 점수 식과 백분위 판정
- `detector/pipeline.py` — 검사 실행
- `tests/test_scores.py`

학습 없이 사진 한 장의 점수를 계산합니다. AeroBlade의 오토인코더 재구성 오차, RDD의 회전·저역통과 디바이어싱, WaRPAD의 DINOv2 고주파 민감도를 같이 봅니다. 점수가 높을수록 실제 사진 쪽입니다.

AI 사진이 아직 없어도 됩니다. 가지고 있는 실제 중고 사진으로 점수 분포를 저장한 뒤, 검사 사진이 그 분포의 아래쪽에 있으면 AI 의심으로 표시합니다.

## 설치

이미 CPU용 PyTorch가 있습니다. 나머지 패키지만 설치하면 됩니다.

```powershell
pip install --upgrade-strategy only-if-needed -r requirements.txt
```

처음 실행할 때 Stable Diffusion 1.5 오토인코더, LPIPS VGG, DINOv2-base 가중치를 받습니다. 합쳐서 1GB 안쪽입니다.

## 데이터 위치

- `data/real` : 실제 중고 사진. 같은 앱에서 받은 사진을 30장 이상.
- `data/query` : 지금 판정할 사진.
- `data/fake` : 나중에 모을 AI 사진. 지금은 비워 둡니다.

사진 원본을 다시 저장하거나 화질을 낮추지 말고, 받은 파일 그대로 넣으세요. 상품 사진만 남기고 앱 화면의 버튼·글자는 잘라 두는 편이 맞습니다.

## 실행

이 폴더(`src/models/model3_TrainingFree`)에서:

```powershell
python detect.py calibrate --real data/real
python detect.py predict --input data/query
```

기준은 `outputs/real_reference.json`, 판정은 `outputs/predictions.csv`에 저장됩니다. 엑셀에서 바로 열 수 있게 UTF-8 BOM으로 저장합니다.

한 장만 보려면:

```powershell
python detect.py predict --input "C:\경로\상품.jpg"
```

AI 사진이 모이면:

```powershell
python detect.py evaluate --real data/real --fake data/fake
```

AUROC가 1에 가까우면 두 폴더가 잘 갈리고, 0.5면 점수만으로는 갈리지 않는 상태입니다.

## 판정

종합 백분위는 실제 사진 기준과 비교한 위치입니다. 점수가 기준 사진들과 같으면 50% 근처가 됩니다.

- 5% 미만: AI 의심
- 5% 이상 20% 미만: 애매
- 20% 이상: 실제에 가까움

이 기준은 논문의 고정 임계값이 아닙니다. 사용자의 실제 사진 분포에서 벗어난 정도입니다. `--low-percentile`, `--mid-percentile`로 바꿀 수 있습니다.

## 점수가 의미하는 것

- `재구성오차_fAE`: Stable Diffusion 오토인코더가 사진을 다시 그릴 때의 LPIPS. AeroBlade에서 AI 이미지는 이 오차가 작습니다.
- `이미지디바이어싱_Simage`: 원본 오차에서 저역통과·90도 회전 오차를 뺀 RDD 점수. 흰 배경 상품처럼 복원이 쉬운 실제 사진이 AI로 오인되는 경우를 줄이려고 만든 항입니다.
- `WaRPAD점수`: DINOv2 특징이 Haar 고주파를 빼도 유지되는 정도. 확산 오토인코더와 다른 종류의 생성 모델을 보완합니다.
- `잠재디바이어싱_Slatent`, `RDD점수`: `--latent`를 켰을 때만 계산합니다. RDD 최종 점수는 `Simage × Slatent²`입니다.

기본 `fast`는 SD 1.5 오토인코더 하나와 DINOv2-base(변 448)입니다. CPU에서 장당 수십 초 정도를 보세요.

논문에 가깝게 돌리려면 GPU에서:

```powershell
python detect.py calibrate --preset paper --latent --real data/real
python detect.py predict --preset paper --latent --input data/query
```

`--preset paper`는 SD 1.4와 SD 2-base 오토인코더를 평균하고, DINOv2-large와 변 896을 씁니다. `--latent`는 확산 UNet까지 불러오므로 CPU에서는 장당 수 분이 걸릴 수 있습니다. 논문의 동기 실험은 노이즈 표본 16개이고, 여기서 기본값은 4개입니다. `--noise-samples 16`으로 맞출 수 있습니다.

논문 하이퍼파라미터 기본값: 회전 계수 0.5, 가우시안 커널 3, 표준편차 0.8, 확산 시각 0.05, WaRPAD 섭동 세기 0.1, LPIPS VGG 두 번째 층.

## 읽을 때 주의할 점

재구성 점수는 큰 사진의 한가운데 512×512만 봅니다. 상품이 구석에 있으면 그 영역이 점수에 들어가지 않습니다. WaRPAD는 사진 전체를 정사각으로 맞춰 봅니다.

실제 기준 사진과 검사 사진의 촬영 앱, 압축, 해상도가 다르면 실제 사진도 AI 의심으로 나올 수 있습니다. 기준 폴더를 검사하려는 중고 앱의 실제 사진으로 채우세요.

이 구현은 논문의 점수 정의를 따른 검사기입니다. MiniSD는 공개 체크포인트 구성이 달라 앙상블에서 빼 두었습니다. 공식 저장소의 캐시된 수치와 소수점까지 같지는 않습니다.
