
import os
import re
import zipfile
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.nn import LGConv
except Exception as e:
    st.error("缺少 torch-geometric，請先執行：pip install -r requirements.txt")
    st.stop()

# ==========================================================
# GraphFlix Web App
# Based on the uploaded GraphFlix v2 Python project
# ==========================================================

st.set_page_config(
    page_title="GraphFlix",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "saved_model"
DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

CONFIG = {
    "embedding_dim": 64,
    "num_layers": 3,
    "learning_rate": 0.001,
    "epochs": 30,
    "batch_size": 2048,
    "top_k": 10,
}

MODEL_PATH = MODEL_DIR / "graphflix_lightgcn.pth"
ZIP_PATH = DATASET_DIR / "ml-100k.zip"
EXTRACT_PATH = DATASET_DIR / "ml-100k"

def get_tmdb_api_key():
    """Read TMDb key from Streamlit Cloud Secrets first, then local environment."""
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("TMDB_API_KEY", "").strip()

TMDB_API_KEY = get_tmdb_api_key()
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

@st.cache_data
def load_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    if not ZIP_PATH.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        ZIP_PATH.write_bytes(r.content)

    if not EXTRACT_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATASET_DIR)

    ratings = pd.read_csv(
        EXTRACT_PATH / "u.data",
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        EXTRACT_PATH / "u.item",
        sep="|",
        encoding="latin1",
        header=None,
        usecols=[0, 1]
    )
    movies.columns = ["movieId", "title"]

    ratings = ratings.drop_duplicates().reset_index(drop=True)
    movies = movies.drop_duplicates().reset_index(drop=True)

    # Match the uploaded project's LabelEncoder behavior.
    user_ids = sorted(ratings["userId"].unique())
    movie_ids = sorted(ratings["movieId"].unique())
    user_map = {v: i for i, v in enumerate(user_ids)}
    movie_map = {v: i for i, v in enumerate(movie_ids)}

    ratings["user_idx"] = ratings["userId"].map(user_map)
    ratings["movie_idx"] = ratings["movieId"].map(movie_map)

    movie_info = ratings[["movieId", "movie_idx"]].drop_duplicates().merge(
        movies, on="movieId", how="left"
    ).sort_values("movie_idx").reset_index(drop=True)

    return ratings, movie_info, len(user_ids), len(movie_ids)

class GraphFlixLightGCN(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        x = torch.cat(
            [self.user_embedding.weight, self.movie_embedding.weight],
            dim=0
        )
        embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embeddings.append(x)
        final = torch.stack(embeddings, dim=0).mean(dim=0)
        return final[:self.num_users], final[self.num_users:]

def build_graph(train_ratings, num_users):
    u = train_ratings["user_idx"].values
    m = train_ratings["movie_idx"].values + num_users
    edge = np.vstack([
        np.concatenate([u, m]),
        np.concatenate([m, u])
    ])
    return torch.LongTensor(edge)

def prepare_splits(ratings):
    train_list, val_list, test_list = [], [], []
    for user in ratings["user_idx"].unique():
        data = ratings[ratings["user_idx"] == user]
        if len(data) < 5:
            train_list.append(data)
            continue
        train, temp = train_test_split(
            data, test_size=0.2, random_state=42
        )
        val, test = train_test_split(
            temp, test_size=0.5, random_state=42
        )
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    return train, val, test

def sample_negative(user, positives, num_movies, rng):
    while True:
        x = int(rng.integers(0, num_movies))
        if x not in positives[user]:
            return x

def train_model(ratings, num_users, num_movies, epochs):
    train, val, _ = prepare_splits(ratings)
    edge_index = build_graph(train, num_users)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphFlixLightGCN(
        num_users, num_movies,
        CONFIG["embedding_dim"], CONFIG["num_layers"]
    ).to(device)
    edge_index = edge_index.to(device)

    positives = defaultdict(set)
    for row in train.itertuples():
        positives[row.user_idx].add(row.movie_idx)

    users = []
    pos = []
    neg = []
    rng = np.random.default_rng(42)
    for row in train.itertuples():
        users.append(row.user_idx)
        pos.append(row.movie_idx)
        neg.append(sample_negative(row.user_idx, positives, num_movies, rng))

    users = torch.LongTensor(users)
    pos = torch.LongTensor(pos)
    neg = torch.LongTensor(neg)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    progress = st.progress(0)
    status = st.empty()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        user_emb, movie_emb = model(edge_index)
        pos_score = (user_emb[users.to(device)] * movie_emb[pos.to(device)]).sum(1)
        neg_score = (user_emb[users.to(device)] * movie_emb[neg.to(device)]).sum(1)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        progress.progress((epoch + 1) / epochs)
        status.write(f"Epoch {epoch+1}/{epochs}　Loss: {loss.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_movies": num_movies,
        "embedding_dim": CONFIG["embedding_dim"],
        "num_layers": CONFIG["num_layers"],
        "train_user_ids": train["user_idx"].unique().tolist(),
    }
    torch.save(checkpoint, MODEL_PATH)

    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)

    return model, user_emb.cpu(), movie_emb.cpu(), train, edge_index.cpu()

@st.cache_resource
def load_or_train_model(_ratings, num_users, num_movies):
    if MODEL_PATH.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model = GraphFlixLightGCN(
            checkpoint["num_users"],
            checkpoint["num_movies"],
            checkpoint["embedding_dim"],
            checkpoint["num_layers"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        train, _, _ = prepare_splits(_ratings)
        edge_index = build_graph(train, num_users).to(device)
        with torch.no_grad():
            user_emb, movie_emb = model(edge_index)
        return model, user_emb.cpu(), movie_emb.cpu(), train, edge_index.cpu(), "已載入模型"

    return None

def recommend(user_id, top_k, ratings, movie_info, user_emb, movie_emb):
    user_vector = user_emb[user_id]
    scores = torch.matmul(movie_emb, user_vector).clone()

    watched = ratings[ratings["user_idx"] == user_id]["movie_idx"].tolist()
    if watched:
        scores[watched] = -float("inf")

    k = min(top_k, len(scores) - len(watched))
    values, indices = torch.topk(scores, k=k)

    score_min = float(values.min())
    score_max = float(values.max())

    rows = []
    for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), 1):
        row = movie_info[movie_info["movie_idx"] == idx]
        if row.empty:
            continue
        row = row.iloc[0]
        if score_max - score_min < 1e-8:
            confidence = 100.0
        else:
            confidence = (float(score) - score_min) / (score_max - score_min) * 100

        rows.append({
            "rank": rank,
            "movieId": int(row["movieId"]),
            "movie_idx": int(idx),
            "title": row["title"],
            "score": float(score),
            "confidence": float(confidence),
        })
    return rows

@st.cache_data(ttl=3600)
def tmdb_search(title):
    if not TMDB_API_KEY:
        return None

    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()
    year_match = re.search(r"\((\d{4})\)", str(title))
    params = {
        "api_key": TMDB_API_KEY,
        "query": clean,
        "language": "zh-TW",
        "include_adult": False,
    }
    if year_match:
        params["year"] = year_match.group(1)

    try:
        r = requests.get(f"{TMDB_BASE}/search/movie", params=params, timeout=15)
        if r.status_code != 200:
            return None
        results = r.json().get("results", [])
        return results[0] if results else None
    except requests.RequestException:
        return None

@st.cache_data(ttl=3600)
def tmdb_details(tmdb_id):
    if not TMDB_API_KEY or not tmdb_id:
        return None
    try:
        r = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY, "language": "zh-TW"},
            timeout=15
        )
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None

def enrich(movie):
    data = tmdb_search(movie["title"])
    if not data:
        return movie

    details = tmdb_details(data.get("id"))
    details = details or data
    poster = details.get("poster_path")
    movie = dict(movie)
    movie.update({
        "tmdb_id": details.get("id"),
        "tmdb_title": details.get("title"),
        "overview": details.get("overview"),
        "release_date": details.get("release_date"),
        "tmdb_rating": details.get("vote_average"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "poster_url": f"{TMDB_IMAGE_BASE}/w500{poster}" if poster else None,
        "backdrop_url": f"{TMDB_IMAGE_BASE}/w1280{details.get('backdrop_path')}"
            if details.get("backdrop_path") else None,
    })
    return movie

# ---------------- UI ----------------
st.title("🎬 GraphFlix")
st.caption("LightGCN + MovieLens + TMDb 電影推薦系統｜雲端部署版")

with st.sidebar:
    st.header("⚙️ 設定")
    top_k = st.slider("推薦數量", 5, 20, CONFIG["top_k"])
    user_id = st.number_input("User ID", min_value=0, value=0, step=1)
    st.divider()
    st.write("TMDb API：", "✅ 已設定" if TMDB_API_KEY else "⚠️ 未設定")
    st.caption("請使用環境變數 TMDB_API_KEY，不要把 API Key 寫死在程式中。")

try:
    ratings, movie_info, num_users, num_movies = load_movielens()
except Exception as e:
    st.error(f"MovieLens 資料載入失敗：{e}")
    st.stop()

if user_id >= num_users:
    st.error(f"User ID 必須介於 0 ～ {num_users - 1}")
    st.stop()

st.info(f"MovieLens：{num_users:,} 位使用者、{num_movies:,} 部電影、{len(ratings):,} 筆評分")

if "model_data" not in st.session_state:
    try:
        loaded = load_or_train_model(ratings, num_users, num_movies)
        if loaded is None:
            st.warning("雲端尚未找到模型。第一次部署請按下方按鈕訓練；雲端重新啟動後若模型暫存被清除，可能需要再次訓練。")
            if st.button("🚀 初始化 / 訓練 GraphFlix 模型", type="primary"):
                with st.spinner("第一次訓練可能需要一些時間，請稍候…"):
                    result = train_model(
                        ratings, num_users, num_movies,
                        CONFIG["epochs"]
                    )
                    st.session_state["model_data"] = result + ("新訓練模型",)
                st.success("模型訓練完成！")
                st.rerun()
        else:
            st.session_state["model_data"] = loaded
    except Exception as e:
        st.error(f"模型載入/訓練失敗：{e}")
        st.stop()

if "model_data" in st.session_state:
    model, user_emb, movie_emb, train, edge_index, model_status = st.session_state["model_data"]

    tab1, tab2 = st.tabs(["🍿 個人推薦", "🔎 電影搜尋"])

    with tab1:
        st.subheader(f"為 User {user_id} 推薦電影")
        if st.button("✨ 產生推薦", type="primary"):
            with st.spinner("正在計算推薦…"):
                recs = recommend(
                    int(user_id), top_k, ratings, movie_info,
                    user_emb, movie_emb
                )
                enriched = [enrich(x) for x in recs]
                st.session_state["recs"] = enriched

        if "recs" in st.session_state:
            recs = st.session_state["recs"]
            for movie in recs:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([1, 2, 4])
                    with c1:
                        if movie.get("poster_url"):
                            st.image(movie["poster_url"], use_container_width=True)
                        else:
                            st.markdown("🎞️")
                    with c2:
                        st.markdown(f"### #{movie['rank']} {movie['title']}")
                        st.metric("Confidence", f"{movie['confidence']:.1f}%")
                        st.write(f"GraphFlix Score：{movie['score']:.4f}")
                    with c3:
                        st.write(f"TMDb：{movie.get('tmdb_title') or '未找到'}")
                        st.write(f"評分：{movie.get('tmdb_rating') or 'N/A'}")
                        st.write(f"上映：{movie.get('release_date') or '未知'}")
                        genres = movie.get("genres") or []
                        st.write("類型：" + (", ".join(genres) if genres else "N/A"))
                        if movie.get("overview"):
                            st.write(movie["overview"])

    with tab2:
        st.subheader("🔎 TMDb 電影搜尋")
        keyword = st.text_input("輸入電影名稱或關鍵字")
        if keyword:
            if not TMDB_API_KEY:
                st.warning("要使用 TMDb 搜尋，請先設定 TMDB_API_KEY。")
            else:
                with st.spinner("搜尋中…"):
                    params = {
                        "api_key": TMDB_API_KEY,
                        "query": keyword,
                        "language": "zh-TW",
                        "include_adult": False
                    }
                    try:
                        r = requests.get(
                            f"{TMDB_BASE}/search/movie",
                            params=params,
                            timeout=15
                        )
                        results = r.json().get("results", []) if r.status_code == 200 else []
                    except requests.RequestException:
                        results = []

                if not results:
                    st.info("找不到相關電影。")
                else:
                    for m in results[:10]:
                        with st.container(border=True):
                            cols = st.columns([1, 4])
                            poster = m.get("poster_path")
                            with cols[0]:
                                if poster:
                                    st.image(
                                        f"{TMDB_IMAGE_BASE}/w342{poster}",
                                        use_container_width=True
                                    )
                            with cols[1]:
                                st.markdown(f"### {m.get('title', '未知')}")
                                st.write(f"原始片名：{m.get('original_title', '未知')}")
                                st.write(f"上映日期：{m.get('release_date') or '未知'}")
                                st.write(f"TMDb 評分：{m.get('vote_average', 0):.1f}")
                                if m.get("overview"):
                                    st.write(m["overview"])

st.divider()
st.caption("GraphFlix v2｜推薦核心：LightGCN｜資料：MovieLens｜電影資訊：TMDb")
