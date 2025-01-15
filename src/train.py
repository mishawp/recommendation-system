import src.config
import pandas as pd
import pickle
import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
from datetime import datetime
from typing import Literal


def save_model(
    name: str,
    model: BaseEstimator,
    path: str = "./models/",
    method: Literal["catboost", "pickle", "joblib"] = "pickle",
) -> None:
    """сохраняет модель в директории path с именем name + текущее дата_время + .*

    Args:
        name (str): название модели
        model (BaseEstimator): модель
        path (str, optional): путь к директории. Defaults to "./models/".
        method (Literal[&quot;catboost&quot;, &quot;pickle&quot;, &quot;joblib&quot;], optional): способ сохранения. Defaults to "pickle".
    """
    model_path = rf"{path}{name}{datetime.now().strftime('%y%m%d%H%M%S')}"
    if method == "pickle":
        with open(f"{model_path}.pkl", "wb") as f:
            pickle.dump(model, f)
    elif method == "catboost":
        model.save_model(f"{model_path}.cbm")
    elif method == "joblib":
        joblib.dump(model, f"{model_path}.pkl")
    else:
        ValueError(f"There is no such method to save as {method}")


def load_model(
    name: str,
    path: str = "./models/",
    method: Literal["catboost", "pickle", "joblib"] = "pickle",
) -> BaseEstimator:
    """загружает модель из файла

    Args:
        name (str): имя файла с расширением
        path (str, optional): путь к директории файла. Defaults to "./models/".
        method (Literal[&quot;catboost&quot;, &quot;pickle&quot;, &quot;joblib&quot;], optional): способ загрузки. Defaults to "pickle".

    Returns:
        BaseEstimator: загруженная модель
    """

    model_path = rf"{path}{name}"

    if method == "pickle":
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    elif method == "catboost":
        model = CatBoostClassifier()
        model.load_model(model_path)
    elif method == "joblib":
        model = joblib.load(model_path)
    else:
        ValueError(f"There is no such method to save as {method}")

    return model


def common_hitrate(
    X_test: pd.DataFrame, y_test: pd.Series, y_proba: pd.Series
) -> float:
    """Считает общий хитрейт. Для всех юзеров отбирается топ 5 рекомендаций, далее считается HitRate@5 для каждого юзера. Функция отношение юзеров с HitRate@5 == 1 ко всем юзером"""
    temp = pd.concat([X_test["user_id"], y_test, y_proba], axis=1)
    temp.columns = ["user_id", "y_test", "y_proba"]
    hitrates = temp.groupby("user_id", group_keys=False).apply(
        lambda group: group.nlargest(5, columns="y_proba")["y_test"].any()
    )
    return hitrates.sum() / hitrates.shape[0]


def model_score(X_test: pd.DataFrame, y_test: pd.Series, y_proba: pd.Series) -> None:
    """Выводит три метрики ROC-AUC, PR-AUC, common_hitrate"""
    print("roc_auc_score:", roc_auc_score(y_test, y_proba))
    print("average_precision_score:", average_precision_score(y_test, y_proba))
    print(
        "hitrate:",
        common_hitrate(X_test, y_test, pd.Series(y_proba, index=y_test.index)),
    )
