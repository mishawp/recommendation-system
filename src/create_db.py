import src.config
import gc
import pandas as pd
from typing import Any, Dict, Optional, Literal
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.data import CustomTransformer

TABLE_USERS = "mihail_valiev_gjj7899_users"
TABLE_POSTS = "mihail_valiev_gjj7899_posts"
TABLE_LIKES = "mihail_valiev_gjj7899_likes"


def construct_features(
    create_users_features: bool = True,
    create_posts_features: bool = True,
    create_likes_table: bool = True,
    save_path: Optional[str] = "./data/processed/",
) -> Any:
    """
    1) с помощью CustomTransformer посчитаем фичи для юзеров и закинем их в TABLE_USERS
    2) посчитаем количество символов в posts и закинем их в TABLE_POSTS
    3) чтобы в будущем фильтровать пары user_id post_id, у которых action == "like", отберем все такие пары из feed_data и занесем их в TABLE_LIKES

    Args:
        create_users_features (bool, optional): создать фичи для юзеров. Defaults to True.
        create_posts_features (bool, optional): создать фичи для постов. Defaults to True.
        create_likes_table (bool, optional): создать таблицу пар user_id, post_id у которых action = like. Defaults to True.
        save_path (Optional[str], optional): путь к директории, куда сохранить полученные фичи. Defaults to "./data/processed/".

    Returns:
        Any: может быть пустой tuple или tuple из соответствующих датафреймов
    """
    table_users_df, table_posts_df, table_likes_df = None, None, None
    # -----------------------------------------------------------
    if create_likes_table:
        feed_data = pd.read_csv("./data/raw/feed_data.csv", parse_dates=["timestamp"])
        table_likes_df = (
            feed_data.query("target == 1")[["user_id", "post_id"]]
            .drop_duplicates()
            .copy()
        )
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    if create_users_features:
        X = feed_data.drop("target", axis=1)
        y = feed_data["target"]
        custom_trans = CustomTransformer(drop_cols=[], text_length=False)
        custom_trans.fit(X, y, copy=False)
        del feed_data, X, y
        gc.collect()  # Освобождаем память
        users = pd.read_csv("./data/raw/users.csv", index_col="user_id")
        users.drop("country", axis=1, inplace=True)
        table_users_df = users.join(custom_trans.new_features_user)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    if create_posts_features:
        posts = pd.read_csv("./data/raw/posts.csv", index_col="post_id")
        posts["length"] = posts["text"].str.len()
        table_posts_df = posts.drop("text", axis=1)
    # -----------------------------------------------------------

    if save_path is not None:
        table_likes_df.to_csv(f"{save_path}table_likes.csv", index=False)
        table_users_df.to_csv(f"{save_path}table_users.csv", index=True)
        table_posts_df.to_csv(f"{save_path}table_posts.csv", index=True)

    return tuple(
        filter(
            lambda x: x is not None, (table_users_df, table_posts_df, table_likes_df)
        )
    )


def push_to_db(
    tables: Dict[str, pd.DataFrame],
    engine: Engine,
    if_exists: Literal["fail", "replace", "append"] = "replace",
) -> None:
    """Загружает датафреймы в базу данных. Индексы должны быть сброшены.

    Args:
        tables (Dict[str, pd.DataFrame]): слова ключ - название таблицы, значение - датафрейм
        engine (Engine): sqlalchemy engine
        if_exists (Literal[&quot;fail&quot;, &quot;replace&quot;, &quot;append&quot;], optional): что делать, если таблица уже существует. Defaults to "replace".
    """
    for df in tables.values():
        if not isinstance(df.index, pd.RangeIndex):
            raise ValueError("Индекс DataFrame должен быть RangeIndex.")
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists=if_exists, index=False)


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features() -> pd.DataFrame:
    query = f"""--sql
    SELECT *
    FROM {TABLE_USERS};
    """
    table_users_df = batch_load_sql(query)

    return table_users_df


if __name__ == "__main__":
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )

    tables_names = [TABLE_USERS, TABLE_POSTS, TABLE_LIKES]
    tables_df = (
        pd.read_csv("./data/processed/table_users.csv"),
        pd.read_csv("./data/processed/table_posts.csv"),
        pd.read_csv("./data/processed/table_likes.csv"),
    )
    tables = dict(zip(tables_names, tables_df))
    push_to_db(tables, engine)
