import os
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
    model_path = get_model_path("/my/super/path")
    # with open(model_path, "rb") as file:
    #     model = pickle.load(file)
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


def load_features() -> pd.DataFrame:
    query = f"""--sql
    SELECT *
    FROM {TABLE_USERS};
    """
    table_users_df = batch_load_sql(query)

    return table_users_df
