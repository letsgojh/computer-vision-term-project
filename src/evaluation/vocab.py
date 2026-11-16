"""결과표에 쓰는 이름. 데이터 규약의 영문 값과 발표용 한글을 연결한다."""

from src.common.schema import POSTPROCESSES

POSTPROCESS_KO = {
    "original": "원본",
    "kakao": "카톡 압축",
    "resize": "리사이즈",
    "recapture": "재촬영",
}
POSTPROCESS_COLUMNS = [POSTPROCESS_KO[name] for name in POSTPROCESSES]

SCOPE_KO = {"all": "전체", "A": "A", "B": "B", "C": "C"}
NOTION_SCOPES = ("A", "B", "C")
ALL_SCOPES = ("all", "A", "B", "C")

MODEL_KO = {"model1": "모델 1", "model2": "모델 2"}

# 유형별 칸은 그 유형의 가짜와 진짜만 본다. 캡션에서 이 정의를 반복한다.
SCOPE_NOTE = (
    "유형 A·B·C 칸은 그 유형의 가짜와 진짜만 비교한 값입니다. "
    "다른 유형의 가짜는 그 칸에 넣지 않습니다. "
    "전체 칸은 모든 가짜를 합친 값입니다."
)

METRIC_HELP = [
    ("Accuracy", "전체 중 맞힌 비율"),
    ("Precision", "가짜라고 한 것 중 실제로 가짜인 비율 (오탐이 적은가)"),
    ("Recall", "실제 가짜 중 잡아낸 비율 (놓치는 게 적은가)"),
    ("F1", "Precision과 Recall의 균형"),
    ("AUC", "판단 기준값과 상관없이 진짜와 가짜를 얼마나 잘 구분하는가"),
    ("Specificity", "실제 진짜 중 진짜라고 맞힌 비율"),
    ("Balanced Accuracy", "Recall과 Specificity의 평균. 진짜·가짜 장수가 달라도 Accuracy가 한쪽으로 끌리는지 확인"),
    ("AP", "정밀도-재현율 곡선 면적. 가짜 장수가 적어도 순위가 맞는지"),
    ("Brier", "확률과 정답의 평균 제곱 오차. 낮을수록 확률이 정답에 가까움"),
    ("ECE", "0.8이라고 한 묶음이 실제로 약 80% 가짜인지. 낮을수록 확률이 정직함"),
    ("확률 간격", "가짜 평균 확률에서 진짜 평균 확률을 뺀 값. 두 집단이 벌어져 있는지"),
]

META_FIELDS = ["pen", "paper", "lighting", "background", "phone_model", "object", "photographer"]
META_KO = {
    "pen": "펜",
    "paper": "종이",
    "lighting": "조명",
    "background": "배경",
    "phone_model": "폰",
    "object": "물건",
    "photographer": "촬영자",
}

TYPE_HINT = {
    "A": "메모 글씨를 인페인팅한 유형입니다. 날짜·닉네임이 바뀐 흔적이 약한 샘플을 우선 보강합니다.",
    "B": "제품 영역을 인페인팅한 유형입니다. 물건 경계가 자연스러운 샘플이 더 어렵습니다.",
    "C": "처음부터 생성한 유형입니다. 손글씨 메모, 조명, 배경 조합을 더 다양하게 만듭니다.",
}
POSTPROCESS_HINT = {
    "original": "원본에서도 약합니다. 후처리 이전에 생성 흔적 자체가 약합니다.",
    "kakao": "카톡 압축 뒤에 약해집니다. 압축된 버전을 train에 함께 넣습니다.",
    "resize": "리사이즈 뒤에 약해집니다. 해상도를 낮춘 버전을 train에 함께 넣습니다.",
    "recapture": "재촬영 뒤에 약해집니다. 화면을 다시 찍은 버전을 train에 함께 넣습니다.",
}
