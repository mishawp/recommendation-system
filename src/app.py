import os
from typing import List, Tuple
from fastapi import FastAPI

# from src.schema import PostGet
from schema import PostGet
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier


TABLE_USERS = "mihail_valiev_gjj7899_users"
TABLE_POSTS = "mihail_valiev_gjj7899_posts"
TABLE_LIKES = "mihail_valiev_gjj7899_likes"


def get_model_path(path: str) -> str:
    if (
        os.environ.get("IS_LMS") == "1"
    ):  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = "/workdir/user_input/model"
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path(
        r"C:\Users\MishaV\Desktop\karpov\ML\L22\models\embeddings.cbm"
    )
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


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


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = f"""--sql
    SELECT *
    FROM {TABLE_USERS};
    """
    table_users_df = batch_load_sql(query)
    table_users_df.set_index("user_id", inplace=True)

    query = f"""--sql
    SELECT *
    FROM {TABLE_POSTS};
    """
    table_posts_df = batch_load_sql(query)
    table_posts_df.set_index("post_id", inplace=True)

    query = f"""--sql
    SELECT *
    FROM {TABLE_LIKES};
    """
    table_likes_df = batch_load_sql(query)
    table_likes_df.set_index("user_id", inplace=True)

    return table_users_df, table_posts_df, table_likes_df


def register_routes(
    app: FastAPI,
    model: CatBoostClassifier,
    table_users_df: pd.DataFrame,
    table_posts_df: pd.DataFrame,
    table_likes_df: pd.DataFrame,
):
    @app.get("/post/recommendations/", response_model=List[PostGet])
    def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 10,
    ) -> List[PostGet]:
        liked_posts = table_likes_df.loc[id]["post_id"]

        # Учитываем, что liked_posts может быть скалярным, если у пользователя только один лайк.
        if isinstance(liked_posts, pd.Series):
            not_liked_posts = table_posts_df[~table_posts_df.index.isin(liked_posts)]
        else:
            not_liked_posts = table_posts_df[table_posts_df.index != liked_posts]
        if isinstance(not_liked_posts, pd.Series):
            not_liked_posts = pd.DataFrame([not_liked_posts])

        user = pd.DataFrame(
            [table_users_df.loc[id]] * not_liked_posts.shape[0]
        ).rename_axis("user_id")
        not_liked_posts.reset_index(inplace=True)
        user.reset_index(inplace=True)
        data = pd.concat(
            [user.drop("user_id", axis=1), not_liked_posts.drop("post_id", axis=1)],
            axis=1,
        ).sort_index(axis=1)
        y_proba = model.predict_proba(data.drop("text", axis=1))[:, 1]
        posts = pd.concat(
            [
                not_liked_posts,
                pd.Series(y_proba, index=not_liked_posts.index).rename("y_proba"),
            ],
            axis=1,
        )
        top = posts.nlargest(limit, columns="y_proba")
        result = top[["post_id", "text", "topic"]]

        return [
            PostGet(id=row["post_id"], text=row["text"], topic=row["topic"])
            for _, row in result.iterrows()
        ]


def init_app() -> FastAPI:
    app = FastAPI()
    model = load_models()
    print("Модел загружена")
    table_users_df, table_posts_df, table_likes_df = load_features()
    print("Признаки загружены")

    # Регистрируем маршруты и передаем данные в эндпоинт
    register_routes(app, model, table_users_df, table_posts_df, table_likes_df)
    print("Маршруты загружены")

    return app


app = init_app()
