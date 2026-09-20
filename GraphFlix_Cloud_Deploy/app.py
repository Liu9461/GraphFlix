
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



# ==========================================================
# Accounts / preferences / browsing history
# ==========================================================
import sqlite3
import hashlib
import hmac
import secrets

DB_PATH = BASE_DIR / "graphflix_users.db"

AVAILABLE_GENRES = ["動作", "科幻", "冒險", "動畫", "喜劇", "驚悚", "劇情", "愛情"]
TMDB_GENRE_IDS = {
    "動作": 28, "科幻": 878, "冒險": 12, "動畫": 16,
    "喜劇": 35, "驚悚": 53, "劇情": 18, "愛情": 10749,
}

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        fav_genres TEXT DEFAULT ''
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS viewed(
        username TEXT NOT NULL,
        tmdb_id INTEGER NOT NULL,
        title TEXT,
        viewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(username, tmdb_id)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS favorites(
        username TEXT NOT NULL,
        tmdb_id INTEGER NOT NULL,
        title TEXT,
        added_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(username, tmdb_id)
    )""")
    conn.commit()
    return conn

def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 150_000)
    return salt, digest.hex()

def register_user(username, password, fav_genres):
    username = username.strip()
    if len(username) < 3:
        return False, "帳號至少需要 3 個字元。"
    if len(password) < 4:
        return False, "密碼至少需要 4 個字元。"
    salt, digest = password_hash(password)
    try:
        conn = db()
        conn.execute(
            "INSERT INTO users(username,salt,password_hash,fav_genres) VALUES(?,?,?,?)",
            (username, salt, digest, ",".join(fav_genres)),
        )
        conn.commit()
        conn.close()
        return True, "註冊成功，現在可以登入。"
    except sqlite3.IntegrityError:
        return False, "這個帳號已經被註冊。"

def login_user(username, password):
    conn = db()
    row = conn.execute(
        "SELECT salt,password_hash,fav_genres FROM users WHERE username=?",
        (username.strip(),),
    ).fetchone()
    conn.close()
    if not row:
        return False, None
    _, digest = password_hash(password, row[0])
    if not hmac.compare_digest(digest, row[1]):
        return False, None
    genres = [x for x in (row[2] or "").split(",") if x]
    return True, genres

def save_genres(username, genres):
    conn = db()
    conn.execute("UPDATE users SET fav_genres=? WHERE username=?", (",".join(genres), username))
    conn.commit()
    conn.close()

def add_history(username, tmdb_id, title):
    if not tmdb_id:
        return
    conn = db()
    conn.execute(
        "INSERT OR IGNORE INTO viewed(username,tmdb_id,title) VALUES(?,?,?)",
        (username, int(tmdb_id), title),
    )
    conn.commit()
    conn.close()

def add_favorite(username, tmdb_id, title):
    if not tmdb_id:
        return
    conn = db()
    conn.execute(
        "INSERT OR IGNORE INTO favorites(username,tmdb_id,title) VALUES(?,?,?)",
        (username, int(tmdb_id), title),
    )
    conn.commit()
    conn.close()

def get_user_list(table, username):
    conn = db()
    rows = conn.execute(
        f"SELECT tmdb_id,title FROM {table} WHERE username=? ORDER BY rowid DESC",
        (username,),
    ).fetchall()
    conn.close()
    return rows

@st.cache_data(ttl=1800)
def discover_by_genre(genre_name, page=1):
    gid = TMDB_GENRE_IDS.get(genre_name)
    if not TMDB_API_KEY or not gid:
        return []
    params = {
        "api_key": TMDB_API_KEY,
        "language": "zh-TW",
        "sort_by": "popularity.desc",
        "include_adult": False,
        "include_video": False,
        "with_genres": gid,
        "page": page,
    }
    try:
        r = requests.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=15)
        return r.json().get("results", []) if r.status_code == 200 else []
    except requests.RequestException:
        return []

def tmdb_result_to_card(m):
    poster = m.get("poster_path")
    return {
        "tmdb_id": m.get("id"),
        "tmdb_title": m.get("title") or m.get("name"),
        "title": m.get("title") or m.get("name"),
        "overview": m.get("overview"),
        "release_date": m.get("release_date"),
        "tmdb_rating": m.get("vote_average"),
        "poster_url": f"{TMDB_IMAGE_BASE}/w500{poster}" if poster else None,
        "genres": [],
    }

def render_movie_card(movie, key_prefix, username=None):
    cols = st.columns([1, 2.4])
    with cols[0]:
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
        else:
            st.info("暫無海報")
    with cols[1]:
        title = movie.get("tmdb_title") or movie.get("title") or "未知電影"
        st.subheader(title)
        bits = []
        if movie.get("release_date"):
            bits.append(f"📅 {movie['release_date']}")
        if movie.get("tmdb_rating") is not None:
            bits.append(f"⭐ {float(movie['tmdb_rating']):.1f}")
        if movie.get("confidence") is not None:
            bits.append(f"🎯 推薦信心 {float(movie['confidence']):.1f}%")
        if bits:
            st.caption("　".join(bits))
        if movie.get("genres"):
            st.write("類型：" + "、".join(movie["genres"]))
        overview = movie.get("overview") or "目前沒有中文電影簡介。"
        st.write(overview)
        if username and movie.get("tmdb_id"):
            b1, b2 = st.columns(2)
            if b1.button("👁️ 標記已看", key=f"view_{key_prefix}_{movie['tmdb_id']}"):
                add_history(username, movie["tmdb_id"], title)
                st.toast("已加入觀看紀錄")
            if b2.button("❤️ 收藏", key=f"fav_{key_prefix}_{movie['tmdb_id']}"):
                add_favorite(username, movie["tmdb_id"], title)
                st.toast("已加入收藏")
    st.divider()

def stable_movielens_user(username, num_users):
    # Each account gets a stable seed node in the MovieLens graph.
    return int(hashlib.sha256(username.encode()).hexdigest()[:8], 16) % num_users


# ==========================================================
# UI
# ==========================================================
st.set_page_config(page_title="GraphFlix", page_icon="🎬", layout="wide")
db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "fav_genres" not in st.session_state:
    st.session_state.fav_genres = []
if "batch" not in st.session_state:
    st.session_state.batch = 0

st.title("🎬 GraphFlix")
st.caption("LightGCN + MovieLens + TMDb｜登入・分類・個人化推薦完整網頁版")

# ---------------- Authentication ----------------
if not st.session_state.logged_in:
    st.info("請先登入；第一次使用可以直接建立帳號。")
    login_tab, register_tab = st.tabs(["🔐 登入", "📝 註冊"])

    with login_tab:
        with st.form("login_form"):
            u = st.text_input("帳號")
            p = st.text_input("密碼", type="password")
            submitted = st.form_submit_button("登入", use_container_width=True)
        if submitted:
            ok, genres = login_user(u, p)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = u.strip()
                st.session_state.fav_genres = genres
                st.rerun()
            else:
                st.error("帳號或密碼錯誤。")

    with register_tab:
        with st.form("register_form"):
            new_u = st.text_input("設定帳號")
            new_p = st.text_input("設定密碼", type="password")
            new_p2 = st.text_input("再次輸入密碼", type="password")
            new_genres = st.multiselect(
                "喜歡的電影類型（可複選）",
                AVAILABLE_GENRES,
                default=["動作", "科幻"],
            )
            reg = st.form_submit_button("建立帳號", use_container_width=True)
        if reg:
            if new_p != new_p2:
                st.error("兩次密碼不一致。")
            else:
                ok, msg = register_user(new_u, new_p, new_genres)
                (st.success if ok else st.error)(msg)

    st.caption("雲端展示版的帳號資料儲存在 Streamlit 執行環境；免費雲端重建環境時資料可能重置。")
    st.stop()

username = st.session_state.username

with st.sidebar:
    st.header(f"👤 {username}")
    st.write("偏好：" + ("、".join(st.session_state.fav_genres) if st.session_state.fav_genres else "尚未設定"))
    if TMDB_API_KEY:
        st.success("TMDb API：已設定")
    else:
        st.warning("TMDb API：未設定")
    if st.button("🚪 登出", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.fav_genres = []
        st.rerun()

ratings, movie_info, num_users, num_movies = load_movielens()
st.caption(f"MovieLens：{num_users:,} 位使用者・{num_movies:,} 部電影・{len(ratings):,} 筆評分")

loaded = load_or_train_model(ratings, num_users, num_movies)
if loaded is None:
    st.warning("尚未找到 LightGCN 模型。第一次部署請先初始化。")
    if st.button("🚀 初始化 / 訓練 GraphFlix 模型", type="primary"):
        with st.spinner("正在訓練 LightGCN，請稍候…"):
            train_model(ratings, num_users, num_movies, CONFIG["epochs"])
        st.cache_resource.clear()
        st.success("模型訓練完成！")
        st.rerun()
    st.stop()

model, user_emb, movie_emb, train_df, edge_index, model_status = loaded

home_tab, genre_tab, search_tab, profile_tab = st.tabs(
    ["🍿 個人推薦", "🎭 電影分類", "🔎 電影搜尋", "👤 我的 GraphFlix"]
)

with home_tab:
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("為你推薦")
        st.caption("LightGCN 產生候選電影，再搭配你的電影偏好與觀看紀錄呈現。")
    with c2:
        if st.button("🔄 換一批", use_container_width=True):
            st.session_state.batch += 1
            st.rerun()

    ml_user = stable_movielens_user(username, num_users)
    candidate_count = min(30, 10 + st.session_state.batch * 5)
    recs = recommend(ml_user, candidate_count, ratings, movie_info, user_emb, movie_emb)
    start = (st.session_state.batch * 8) % max(1, len(recs))
    selected = (recs[start:start+8] or recs[:8])

    with st.spinner("正在取得電影資料…"):
        enriched = [enrich(x) for x in selected]

    prefs = set(st.session_state.fav_genres)
    if prefs:
        # Put movies matching a chosen genre first, without discarding LightGCN results.
        zh_to_tmdb = {"動作":"動作","科幻":"科幻","冒險":"冒險","動畫":"動畫",
                      "喜劇":"喜劇","驚悚":"驚悚","劇情":"劇情","愛情":"愛情"}
        def pref_score(m):
            gs = set(m.get("genres") or [])
            return sum(1 for p in prefs if zh_to_tmdb.get(p) in gs)
        enriched.sort(key=pref_score, reverse=True)

    for i, movie in enumerate(enriched):
        render_movie_card(movie, f"rec{i}", username)

with genre_tab:
    st.subheader("🎭 電影分類")
    genre = st.selectbox("選擇電影類型", AVAILABLE_GENRES,
                         index=AVAILABLE_GENRES.index(st.session_state.fav_genres[0])
                         if st.session_state.fav_genres and st.session_state.fav_genres[0] in AVAILABLE_GENRES else 0)
    page = st.number_input("頁數", min_value=1, max_value=20, value=1, step=1)
    if not TMDB_API_KEY:
        st.warning("請先在 Streamlit Secrets 設定 TMDB_API_KEY。")
    else:
        results = discover_by_genre(genre, int(page))
        st.caption(f"{genre}｜本頁 {len(results)} 部")
        for i, m in enumerate(results[:12]):
            render_movie_card(tmdb_result_to_card(m), f"genre{i}", username)

with search_tab:
    st.subheader("🔎 搜尋電影")
    q = st.text_input("輸入電影名稱", placeholder="例如：Inception、玩具總動員")
    if q:
        if not TMDB_API_KEY:
            st.warning("請先設定 TMDB_API_KEY。")
        else:
            try:
                r = requests.get(
                    f"{TMDB_BASE}/search/movie",
                    params={"api_key": TMDB_API_KEY, "query": q, "language": "zh-TW", "include_adult": False},
                    timeout=15,
                )
                results = r.json().get("results", []) if r.status_code == 200 else []
            except requests.RequestException:
                results = []
            if not results:
                st.info("沒有找到符合的電影。")
            for i, m in enumerate(results[:10]):
                render_movie_card(tmdb_result_to_card(m), f"search{i}", username)

with profile_tab:
    st.subheader("👤 我的 GraphFlix")
    new_prefs = st.multiselect(
        "我的喜好類型",
        AVAILABLE_GENRES,
        default=[g for g in st.session_state.fav_genres if g in AVAILABLE_GENRES],
    )
    if st.button("💾 儲存喜好"):
        save_genres(username, new_prefs)
        st.session_state.fav_genres = new_prefs
        st.success("喜好已更新。")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ❤️ 我的收藏")
        favs = get_user_list("favorites", username)
        if favs:
            for _, title in favs[:30]:
                st.write("• " + (title or "未知電影"))
        else:
            st.caption("目前沒有收藏。")
    with c2:
        st.markdown("#### 👁️ 觀看紀錄")
        viewed = get_user_list("viewed", username)
        if viewed:
            for _, title in viewed[:30]:
                st.write("• " + (title or "未知電影"))
        else:
            st.caption("目前沒有觀看紀錄。")

st.caption("GraphFlix｜推薦核心：LightGCN｜資料：MovieLens｜電影資訊：TMDb")
