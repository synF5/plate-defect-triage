"""Tune all 5 models, evaluate on the test set and save them to model/.

Run from the repo root:  python -m plate_triage.fit_all
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from plate_triage.dataset import (
    HOLDOUT_CSV,
    REPO_ROOT,
    SPLIT_SEED,
    TARGET,
    load_plates,
    split_plates,
)
from plate_triage.grading import grade, mcc_interval
from plate_triage.roster import Contender, build_roster

MODEL_DIR = REPO_ROOT / "model"
SCORECARD = MODEL_DIR / "scorecard.json"


def tune(contender: Contender, X, y):
    # grid search with 5-fold CV on MCC, then refit on the full train set
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    search = GridSearchCV(
        contender.pipeline,
        contender.grid,
        scoring="matthews_corrcoef",
        cv=folds,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X, y)

    final = clone(contender.pipeline).set_params(**search.best_params_)
    if contender.slug == "support_vector_machine":
        # probability=True is slow, so only turn it on for the final fit (needed for AUC)
        final.set_params(clf__probability=True)
    final.fit(X, y)
    return final, search


def main(out_dir: Path = MODEL_DIR) -> dict:
    train_part, holdout_part = split_plates(load_plates())
    holdout_part.to_csv(HOLDOUT_CSV, index=False)

    X_train, y_train = train_part.drop(columns=TARGET), train_part[TARGET]
    X_hold, y_hold = holdout_part.drop(columns=TARGET), holdout_part[TARGET]

    out_dir.mkdir(exist_ok=True)
    entries = {}
    for contender in build_roster():
        started = time.perf_counter()
        fitted, search = tune(contender, X_train, y_train)
        joblib.dump(fitted, out_dir / f"{contender.slug}.joblib", compress=3)

        classes = fitted.classes_
        predicted = fitted.predict(X_hold)
        low, high = mcc_interval(y_hold, predicted)
        per_class = f1_score(y_hold, predicted, labels=classes, average=None, zero_division=0)
        train_metrics = grade(
            y_train, fitted.predict(X_train), fitted.predict_proba(X_train), classes
        )

        entries[contender.title] = {
            "slug": contender.slug,
            "holdout": grade(y_hold, predicted, fitted.predict_proba(X_hold), classes),
            "holdout_mcc_95ci": [low, high],
            "holdout_f1_by_fault": dict(zip(classes.tolist(), per_class.tolist())),
            "train_mcc": train_metrics["MCC"],
            "cv_mcc": float(search.best_score_),
            "best_params": search.best_params_,
        }
        print(
            f"{contender.title:<24} cv_mcc={search.best_score_:.3f} "
            f"holdout_mcc={entries[contender.title]['holdout']['MCC']:.3f} "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    scorecard = {
        "train_plates": len(train_part),
        "holdout_plates": len(holdout_part),
        "split_seed": SPLIT_SEED,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "models": entries,
    }
    (out_dir / SCORECARD.name).write_text(json.dumps(scorecard, indent=2))
    return scorecard


if __name__ == "__main__":
    main()
