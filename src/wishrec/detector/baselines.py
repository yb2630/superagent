"""Baselines: prior-only and TF-IDF + one-vs-rest logistic regression."""

from __future__ import annotations

import numpy as np

from .head import fit_linear_head


def fit_prior(Y: np.ndarray, train_idx: np.ndarray):
    """Constant predictor at the training-fold base rate for every category."""
    train_idx = np.asarray(train_idx)
    rates = np.asarray(Y, dtype=np.float64)[train_idx].mean(axis=0)

    def predict(test_idx: np.ndarray) -> np.ndarray:
        return np.broadcast_to(rates, (len(test_idx), len(rates))).copy()

    return predict, {"rates": [round(float(rate), 6) for rate in rates]}


def fit_tfidf_logistic(
    texts: list[str],
    Y: np.ndarray,
    train_idx: np.ndarray,
    *,
    C: float = 1.0,
    seed: int = 20261121,
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_idx = np.asarray(train_idx)
    Y = np.asarray(Y, dtype=np.float64)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform([texts[index] for index in train_idx])
    models: list[LogisticRegression | None] = []
    constants: list[float | None] = []
    for index in range(Y.shape[1]):
        y = Y[train_idx, index]
        if y.min() == y.max():
            models.append(None)
            constants.append(float((y.sum() + 1.0) / (len(y) + 2.0)))
        else:
            model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=seed)
            model.fit(X_train, y)
            models.append(model)
            constants.append(None)

    def predict(test_idx: np.ndarray) -> np.ndarray:
        X_test = vectorizer.transform([texts[index] for index in np.asarray(test_idx)])
        columns = []
        for model, constant in zip(models, constants):
            if model is None:
                columns.append(np.full(X_test.shape[0], float(constant)))
            else:
                columns.append(model.predict_proba(X_test)[:, 1])
        return np.column_stack(columns)

    return predict, {
        "C": C,
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_features": 20000,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_rows": int(X_train.shape[0]),
        "constant_labels": [index for index, model in enumerate(models) if model is None],
    }


def fit_encoder_head(
    embeddings: np.ndarray,
    Y: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int = 20261121,
    epochs: int = 600,
    learning_rate: float = 0.01,
    l2: float = 0.1,
):
    train_idx = np.asarray(train_idx)
    head = fit_linear_head(
        np.asarray(embeddings)[train_idx],
        np.asarray(Y)[train_idx],
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )

    def predict(test_idx: np.ndarray) -> np.ndarray:
        return head.predict_proba(np.asarray(embeddings)[np.asarray(test_idx)])

    return predict, {
        "head": head,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
    }


__all__ = ["fit_encoder_head", "fit_prior", "fit_tfidf_logistic"]
