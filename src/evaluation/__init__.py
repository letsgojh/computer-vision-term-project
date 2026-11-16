"""인증샷 진위 판별 평가."""

__all__ = ["run_evaluation"]


def __getattr__(name):
    if name == "run_evaluation":
        from src.evaluation.run import run_evaluation

        return run_evaluation
    raise AttributeError(name)
