"""Random Forest model implementation."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest_classifier(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
