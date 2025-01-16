"""
sequence:
1) select_data
2) train_test_split
3) pipeline: CustomTransformer, OneHotEncoder, TargetEncoder, model
"""

import src.config
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from category_encoders.one_hot import OneHotEncoder
from dataclasses import dataclass, field

# from collections import Counter
from typing import Any, Literal, Optional

OHE_COLS = ["gender", "exp_group", "os", "source"]
MTE_COLS = ["city", "topic"]


def select_data_csv(
    save_path: Optional[str],
    users_path: str = "./data/raw/users.csv",
    posts_path: str = "./data/raw/posts.csv",
    feed_data_path: str = "./data/raw/feed_data.csv",
    count: int = 4_800_000,
    method: Literal["stratified", "random"] = "stratified",
    seed: int = 42,
) -> pd.DataFrame:
    """_summary_

    Args:
        save_path (str | None): путь, где сохранится выборка
        users_path (str, optional): путь к таблице user_data. Defaults to "./data/raw/users.csv".
        posts_path (str, optional): путь к таблице post_text_df. Defaults to "./data/raw/posts.csv".
        feed_data_path (str, optional): путь к таблице feed_data. Defaults to "./data/raw/feed_data.csv".
        count (int, optional): примерное количество желаемой выборки. Defaults to 4_800_000.
        method (Literal[&quot;stratified&quot;, &quot;random&quot;], optional): метод отбора. Defaults to "stratified".
        seed (int, optional): сид ГПСЧ. Defaults to 42.

    Returns:
        pd.DataFrame: таблица размера примерно "count", которая является объединением user_data, post_text_df и feed_data
    """
    users = pd.read_csv(users_path, index_col="user_id")
    posts = pd.read_csv(posts_path, index_col="post_id")
    feed_data = pd.read_csv(feed_data_path)
    if method == "stratified":
        feed_data_size = feed_data.shape[0]
        coef = count / feed_data_size
        # количество 0 и 1
        feed_data = feed_data.groupby(["user_id", "target"], group_keys=False).apply(
            lambda group: group.sample(
                n=int(np.ceil(coef * group.shape[0])), random_state=seed
            )
        )
    elif method == "random":
        feed_data = feed_data.sample(n=count, random_state=seed)
    else:
        pass
        raise ValueError(f"There is no such method as {method}")

    data = pd.merge(feed_data, users, left_on="user_id", right_index=True)
    data = pd.merge(data, posts, left_on="post_id", right_index=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    if save_path:
        data.to_csv(save_path, index=False)

    return data


def select_data_csv(
    save_path: Optional[str],
    engine: Any,
    method: Literal["stratified", "random"] = "stratified",
    seed: int = 0.42,
    count: int = 4_800_000,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """select data from database

    Args:
        save_path (str | None): if not None, the data will be saved in the path save_path,
        engine (Any): sqlalchemy engine
        seed (int, optional): seed for rng (from 0 to 1). Defaults to 0
        method (Literal[&quot;stratified&quot;, &quot;random&quot;], optional): method of sampling (only "stratified" or "random"). Defaults to "stratified"
        count (int, optional): the function will return approximately the "count" samples. Defaults to 4_800_000
        chunksize (int, optional): chunksize for pd.read_sql

    Returns:
        pd.DataFrame: a table of size ~"count" rows, which is a union of "user_data", "post_text_df" and "feed_data"
    """
    users = pd.read_sql("SELECT * FROM user_data", con=engine, index_col="user_id")
    posts = pd.read_sql("SELECT * FROM post_text_df", con=engine, index_col="post_id")
    if method == "stratified":
        feed_data = select_stratified_sql(seed, engine, count, chunksize)
    elif method == "random":
        feed_data = select_random_sql(seed, engine, count, chunksize)
    else:
        raise ValueError(f"There is no such method as {method}")

    data = pd.merge(feed_data, users, left_on="user_id", right_index=True)
    data = pd.merge(data, posts, left_on="post_id", right_index=True)

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    if save_path:
        data.to_csv(save_path, index=False)

    return data


def select_stratified_sql(
    seed: int, engine: Any, count: int, chunksize: int
) -> pd.DataFrame:
    """типический отбор по столбцу target. Сохраняет пропорции количеств 1 и 0 в target у юзеров"""

    with engine.connect().execution_options(stream_results=True) as connection:
        connection.execute(f"SELECT setseed({seed});")
        feed_data_size = pd.read_sql("SELECT count(*) FROM feed_data", con=engine).iloc[
            0, 0
        ]
        coef = str(count / feed_data_size)
        query = f"""
        --sql
        WITH counts AS (
            --Подсчет 1 и 0 target для каждого юзера в результатирующей выборке
            SELECT
                user_id,
                target,
                ceil(count(*) * {coef}) as count_
            FROM feed_data
            GROUP BY user_id, target
        ),
        ranked_data AS (
            -- Присваиваем каждой строке ранг внутри каждой группы user_id и target
            SELECT
                fd.*,
                ROW_NUMBER() OVER (PARTITION BY fd.user_id, fd.target ORDER BY RANDOM()) AS rn,
                c.count_ -- Присоединяем рассчитанный размер выборки
            FROM feed_data fd
            JOIN counts c
                ON fd.user_id = c.user_id AND fd.target = c.target
        )
        SELECT fd.timestamp, fd.user_id, fd.post_id, fd.action, fd.target
        FROM ranked_data fd
        WHERE rn <= count_;
        """

        chunks = [
            chunk_dataframe
            for chunk_dataframe in pd.read_sql(query, connection, chunksize=chunksize)
        ]

        feed_data = pd.concat(chunks, ignore_index=True)

    return feed_data


def select_random_sql(
    seed: int, engine: Any, count: int, chunksize: int
) -> pd.DataFrame:
    """случайный отбор"""

    with engine.connect().execution_options(stream_results=True) as connection:
        connection.execute(f"SELECT setseed({seed});")
        feed_data_size = pd.read_sql("SELECT count(*) FROM feed_data", con=engine).iloc[
            0, 0
        ]

        query = f"""
        --sql
        SELECT *
        FROM feed_data
        WHERE random() < ({count}::float / {feed_data_size});
        """
        chunks = [
            chunk_dataframe
            for chunk_dataframe in pd.read_sql(
                query,
                con=connection,
                chunksize=chunksize,
            )
        ]

        feed_data = pd.concat(chunks, ignore_index=True)

    return feed_data


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """split data into training and test samples based on timestamp: newer records are added to the test sample."""
    X_ = X.copy()
    y_ = y.copy()

    if not X["timestamp"].is_monotonic_increasing:
        X_.sort_values(by="timestamp", inplace=True)
        y_ = y_[X_.index]

    test_size_int = int(X_.shape[0] * test_size)
    X_train = X_.iloc[:-test_size_int]
    X_test = X_.iloc[-test_size_int:]
    y_train = y_.iloc[:-test_size_int]
    y_test = y_.iloc[-test_size_int:]

    return X_train, X_test, y_train, y_test


@dataclass
class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Args:
        drop_cols (list, optional): columns to be removed from the table. Defaults to ["text", "user_id", "post_id", "timestamp", "country"]
        user_avg_actions_per_day (bool, optional): compute average actions per day for each user. Defaults to True
        user_target_ratio (bool, optional): compute relative frequency of target for each user. Defaults to True
        user_like_ratio (bool, optional): compute relative frequency of 1 in target for each user. Defaults to True
        text_length (bool, optional): add text_length as new feature. Defaults to True
    """

    drop_cols: list = field(
        default_factory=lambda: [
            "text",
            "user_id",
            "post_id",
            "timestamp",
            "country",
            "action",
        ]
    )
    user_avg_actions_per_day: bool = True
    user_target_ratio: bool = True
    user_like_ratio: bool = True
    text_length: bool = True

    def fit(self, X: pd.DataFrame, y: pd.Series, copy: bool = True):
        """fit

        Args:
            X (pd.DataFrame): features
            y (pd.Series): target
            copy (bool, optional): копировать ли данные, перед их обработкой
        """
        if copy:
            X_ = X.copy()
            y_ = y.copy()
        else:
            X_ = X
            y_ = y

        data = pd.concat([X_, y_], axis=1)

        self.new_features_user = pd.DataFrame(index=data["user_id"].unique())

        if any([self.user_target_ratio, self.user_like_ratio]):
            users_grouped = data.groupby("user_id")

        if self.user_avg_actions_per_day:
            self.new_features_user["user_avg_actions_per_day"] = (
                data.groupby(["user_id", data["timestamp"].dt.date])["target"]
                .count()
                .groupby("user_id")
                .mean()
            )
        if self.user_target_ratio:
            self.new_features_user["user_target_ratio"] = (
                users_grouped["target"].count() / data.shape[0]
            )
        if self.user_like_ratio:
            self.new_features_user["user_like_ratio"] = (
                users_grouped["target"].sum() / users_grouped["target"].count()
            )

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        X_ = X.copy()

        if self.new_features_user.size:
            X_ = pd.merge(
                X_,
                self.new_features_user,
                left_on="user_id",
                right_index=True,
                how="left",
            )
        X_.fillna(0, inplace=True)

        if self.text_length:
            X_["length"] = X_["text"].str.len()

        if self.drop_cols:
            X_.drop(self.drop_cols, axis=1, inplace=True)

        return X_.sort_index(axis=1)


def transform_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
):
    """Производит все преобразования над данными. Чиста, чтоб вне пайплайна это делать. Если надо будет перебирать модели, то это может снизить время обучения"""
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    y_train_ = y_train.copy()
    y_test_ = y_test.copy()

    te_encoder = TargetEncoder(cols=MTE_COLS, return_df=True)
    ohe_encoder = OneHotEncoder(cols=OHE_COLS, return_df=True)
    cs_transformer = CustomTransformer()

    cs_transformer.fit(X_train_, y_train_)
    X_train_ = cs_transformer.transform(X_train_)
    X_test_ = cs_transformer.transform(X_test_)

    te_encoder.fit(X_train_, y_train)
    X_train_ = te_encoder.transform(X_train_)
    X_test_ = te_encoder.transform(X_test_)

    ohe_encoder.fit(X_train_, y_train)
    X_train_ = ohe_encoder.transform(X_train_)
    X_test_ = ohe_encoder.transform(X_test_)

    return X_train_, X_test_, y_train_, y_test_


# def select_encoder_columns(data: pd.DataFrame, n_unique: int) -> tuple[pd.Index, pd.Index]:
#     """Returns a list of columns for OneHotEncoding and a list of columns for MeanTargetEncoding

#     Args:
#         data (pd.DataFrame): data
#         n_unique (int): If the unique values in the column are less than this number, the column belongs to the OneHotEncoding list

#     Returns:
#         tuple[pd.Index, pd.Index]: pair of OneHotEncoding list and MeanTargetEncoding list
#     """
#     pass

# def most_popular_words(texts: pd.Series, n: int = 3, sep: str = "\n\n") -> pd.Series:

#     sentences = texts.str.split(fr'(?:[.!?]\s*|{sep})', n=n, regex=True)
#     sentences = sentences.str.join(" ").str.lower()
#     words_arrays = sentences.apply(lambda x: np.array(list(set(re.findall(r'\b\w+\b', x.lower())))))
#     words_count = Counter(np.concatenate(words_arrays.values))
#     return pd.Series(words_count)
