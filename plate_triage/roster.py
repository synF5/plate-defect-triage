"""The 5 models with their preprocessing and parameter grids."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from plate_triage.dataset import SPLIT_SEED
from plate_triage.features import PlateGeometry

BOTH_FEATURE_SETS = {"geometry__derive": [True, False]}


@dataclass(frozen=True)
class Contender:
    title: str
    slug: str  # file name in model/
    pipeline: Pipeline
    grid: dict


def _gaussianised(classifier) -> Pipeline:
    # some columns are very skewed, so apply yeo-johnson first
    return Pipeline(
        [
            ("geometry", PlateGeometry()),
            ("reshape", PowerTransformer(method="yeo-johnson")),
            ("clf", classifier),
        ]
    )


def build_roster() -> list[Contender]:
    return [
        Contender(
            "Logistic Regression",
            "logistic_regression",
            _gaussianised(LogisticRegression(max_iter=5000)),
            {
                **BOTH_FEATURE_SETS,
                "clf__C": [0.1, 0.3, 1, 3, 10, 30],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        Contender(
            "Decision Tree",
            "decision_tree",
            # no scaling needed for a tree
            Pipeline(
                [
                    ("geometry", PlateGeometry()),
                    ("clf", DecisionTreeClassifier(random_state=SPLIT_SEED)),
                ]
            ),
            {
                **BOTH_FEATURE_SETS,
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [6, 9, 12, 16, None],
                "clf__min_samples_leaf": [1, 3, 5, 10],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        Contender(
            "KNN",
            "knn",
            # robust scaling because of outliers, then pick the best k columns
            Pipeline(
                [
                    ("geometry", PlateGeometry()),
                    ("reshape", RobustScaler()),
                    ("keep", SelectKBest(f_classif)),
                    ("clf", KNeighborsClassifier()),
                ]
            ),
            {
                **BOTH_FEATURE_SETS,
                "keep__k": [10, 15, 20, "all"],
                "clf__n_neighbors": [3, 5, 7, 9, 13, 19],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        Contender(
            "Naive Bayes",
            "naive_bayes",
            # Gaussian NB since features are continuous (some negative, so not Multinomial)
            Pipeline(
                [
                    ("geometry", PlateGeometry()),
                    ("reshape", PowerTransformer(method="yeo-johnson")),
                    ("keep", SelectKBest(f_classif)),
                    ("clf", GaussianNB()),
                ]
            ),
            {
                **BOTH_FEATURE_SETS,
                "keep__k": [6, 10, 15, 20, "all"],
                "clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1],
            },
        ),
        Contender(
            "Support Vector Machine",
            "support_vector_machine",
            _gaussianised(SVC(kernel="rbf", random_state=SPLIT_SEED)),
            {
                **BOTH_FEATURE_SETS,
                "clf__C": [1, 3, 10, 30, 100],
                "clf__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    ]
