"""XGBoost model implementations."""

from xgboost import XGBClassifier, XGBRegressor


def build_xgboost_classifier(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
    )


def build_xgboost_regressor(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=random_state,
    )
