import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Shashank Patel — Music Recommendation System", page_icon="🎵")

@st.cache_data
def load_data():
    songs = pd.read_csv("songs.csv")
    ratings = pd.read_csv("ratings.csv")
    return songs, ratings

songs, ratings = load_data()

st.title("🎵 Music Recommendation System")
st.caption("by Aryan Sharma · Content-based + Collaborative (item-item)")

with st.sidebar:
    st.header("About")
    st.write("Built by **Aryan Sharma**")
    st.write("GitHub: [aryansharma6836-max](https://github.com/aryansharma6836-max)")
    st.write("LinkedIn: [Aryan Sharma](https://www.linkedin.com/in/aryan-sharma-b24151254)")
    st.write("Email: aryansharma6836@gmail.com")

@st.cache_data
def build_content_matrix(df: pd.DataFrame):
    tags = (df["artist"].fillna('') + " " + df["genre"].fillna('')).str.lower()
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(tags)
    sim = cosine_similarity(tfidf_matrix)
    return sim

content_sim = build_content_matrix(songs)

@st.cache_data
def build_item_item_sim(songs_df: pd.DataFrame, ratings_df: pd.DataFrame):
    pivot = ratings_df.pivot_table(index="song_id", columns="user_id", values="rating").fillna(0.0)
    pivot = pivot.reindex(songs_df["song_id"], fill_value=0.0)
    sim = cosine_similarity(pivot.values)
    return sim, pivot.index.tolist()

item_sim, item_index_song_ids = build_item_item_sim(songs, ratings)

def get_song_index_by_title(title: str) -> int:
    match = songs.index[songs["title"] == title]
    return int(match[0]) if len(match) else -1

def get_song_index_by_id(song_id: int) -> int:
    match = songs.index[songs["song_id"] == song_id]
    return int(match[0]) if len(match) else -1

def item_similar_indices(song_idx: int, top_k: int = 10):
    song_id = int(songs.loc[song_idx, "song_id"])
    try:
        item_row = item_index_song_ids.index(song_id)
    except ValueError:
        return []
    sims = item_sim[item_row]
    order = np.argsort(sims)[::-1]
    out = []
    for j in order:
        sid = item_index_song_ids[j]
        idx = get_song_index_by_id(sid)
        if idx != -1 and idx != song_idx:
            out.append((idx, float(sims[j])))
    return out[:top_k]

def content_similar_indices(song_idx: int, top_k: int = 10):
    sims = content_sim[song_idx]
    order = np.argsort(sims)[::-1]
    out = []
    for j in order:
        if j == song_idx:
            continue
        out.append((j, float(sims[j])))
    return out[:top_k]

col1, col2 = st.columns([3,1])
with col1:
    mode = st.radio("Choose input type", ["Pick a song", "Type an artist"], horizontal=True)
with col2:
    top_k = st.number_input("How many recommendations?", min_value=3, max_value=15, value=5, step=1)

alpha = st.slider("Blend: Content vs Collaborative", 0.0, 1.0, 0.6, 0.05,
                  help="1.0 = only content-based, 0.0 = only collaborative")

if mode == "Pick a song":
    title = st.selectbox("Select your favorite song", sorted(songs["title"].unique().tolist()))
    query_idx = get_song_index_by_title(title)
    seed_text = f"Seed song: **{title}** by **{songs.loc[query_idx, 'artist']}**"
else:
    artist = st.text_input("Type an artist name", value="Ed Sheeran")
    mask = songs["artist"].str.lower().str.contains(artist.strip().lower())
    if mask.any():
        query_idx = int(songs[mask].index[0])
    else:
        query_idx = 0
    title = songs.loc[query_idx, "title"]
    seed_text = f"Artist typed: **{artist}** → using seed song **{title}**"

if st.button("Recommend 🎯"):
    st.markdown(seed_text)
    cands_c = content_similar_indices(query_idx, top_k=50)
    cands_i = item_similar_indices(query_idx, top_k=50)

    scores = {}
    for idx, s in cands_c:
        scores[idx] = scores.get(idx, 0.0) + alpha * s
    for idx, s in cands_i:
        scores[idx] = scores.get(idx, 0.0) + (1.0 - alpha) * s

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    rows = []
    count = 0
    for idx, sc in ranked:
        if idx == query_idx:
            continue
        rows.append({
            "Song": songs.loc[idx, "title"],
            "Artist": songs.loc[idx, "artist"],
            "Genre": songs.loc[idx, "genre"],
            "Year": int(songs.loc[idx, "year"]),
            "Score": round(float(sc), 4)
        })
        count += 1
        if count >= top_k:
            break

    if rows:
        st.subheader("Recommended for you")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Not enough data to recommend. Try a different seed.")
else:
    st.write("👆 Pick a song or type an artist, then press **Recommend**.")

st.markdown(
    """
    <hr/>
    <p style="text-align:center;">
    Made with ❤️ by <b>Aryan Sharma</b>
    </p>
    """,
    unsafe_allow_html=True
)
