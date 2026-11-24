"""Constants for the frozen-encoder five-label silence detector.

The positive class for a category is "the category has non-empty content in the full
specification L0", i.e. the wish omits requirement information that the stakeholder can
supply.  A field the source does not contain is not a positive: asking about it can only
return "That is not specified".

Labels are trained on the 500 ``train`` tasks only.  The 200 evolution, 300 held-out and
100 pilot tasks are never used for fitting, calibration, threshold selection or model
selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

#: Five labels, in the milestone order.
LABELS = (
    "intent",
    "signature_types",
    "examples",
    "constraints",
    "error_behaviour",
)

LABEL_DISPLAY = {
    "intent": "intent",
    "signature_types": "signature and types",
    "examples": "examples",
    "constraints": "constraints and edge cases",
    "error_behaviour": "error behaviour",
}

#: The frozen encoder is Qwen3-Embedding-0.6B, whose hidden size is 1024.
ENCODER_NAME = "Qwen/Qwen3-Embedding-0.6B"
ENCODER_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
ENCODER_DIM = 1024
ENCODER_BATCH_SIZE = 16

#: Head and evaluation constants.
DETECTOR_SEED = 20261121
CV_FOLDS = 5
INNER_CALIBRATION_FOLDS = 5
#: L2 coefficient on the head weights; selected on inner CV folds only.
ENCODER_L2_CANDIDATES = (0.03, 0.1, 0.3)
TFIDF_C_CANDIDATES = (1.0, 10.0)
HEAD_EPOCHS = 600
HEAD_LEARNING_RATE = 0.01

#: H3 question policy: predicted presence x utility, calibrated to 2-3 questions/task.
UTILITY = {
    "intent": 0.0,
    "signature_types": 0.5,
    "examples": 0.2,
    "constraints": 1.0,
    "error_behaviour": 1.0,
}
UTILITY_RATIONALE = {
    "intent": "already stated in the wish; asking repeats it",
    "signature_types": "the stub is untyped, so exact types are missing",
    "examples": "three concrete visible tests are shown in every loop condition",
    "constraints": "not recoverable from the wish, stub or visible tests",
    "error_behaviour": "not recoverable from the wish, stub or visible tests",
}
TARGET_MEAN_QUESTIONS = 2.5
QUESTION_COUNT_RANGE = (2, 3)

QUESTION_TEMPLATES = {
    "intent": "What is the core behaviour the function should implement?",
    "signature_types": ("What are the exact function signature, parameter types, and return type?"),
    "examples": ("Can you give a concrete example: an input and the exact expected output?"),
    "constraints": (
        "What constraints and edge cases should the implementation handle (for example "
        "empty inputs, duplicates, ordering, or bounds)?"
    ),
    "error_behaviour": (
        "What should happen on invalid or unexpected input — which exceptions should be "
        "raised, or what should be returned?"
    ),
}

DETECTOR_RUN_ID = "r1"
RESULTS_DETECTOR_DIR = "results/detector"


def utility_vector(labels: tuple[str, ...] = LABELS) -> np.ndarray:
    """Return the question utility of each label, in ``labels`` order, as float64."""
    import numpy as np

    return np.asarray([float(UTILITY[label]) for label in labels], dtype=np.float64)


__all__ = [
    "CV_FOLDS",
    "DETECTOR_RUN_ID",
    "DETECTOR_SEED",
    "ENCODER_BATCH_SIZE",
    "ENCODER_DIM",
    "ENCODER_L2_CANDIDATES",
    "ENCODER_NAME",
    "ENCODER_REVISION",
    "HEAD_EPOCHS",
    "HEAD_LEARNING_RATE",
    "INNER_CALIBRATION_FOLDS",
    "LABELS",
    "LABEL_DISPLAY",
    "QUESTION_COUNT_RANGE",
    "QUESTION_TEMPLATES",
    "TARGET_MEAN_QUESTIONS",
    "TFIDF_C_CANDIDATES",
    "UTILITY",
    "UTILITY_RATIONALE",
    "utility_vector",
]
