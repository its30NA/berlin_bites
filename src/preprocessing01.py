"""
preprocessing.py

Extended preprocessing pipeline for Berlin Bites reviews.

Features:
- Date parsing, deduplication, explode, plotting (existing functionality)
- Review cleaning (newline removal, 'More' removal, HTML cleanup)
- Language detection
- Sentence splitting
- Normalization (lowercase, lemmatization via spaCy, stopword removal)
- Sentiment scoring (TextBlob, fallback to VADER if available)
- Embeddings (sentence-transformers; safe fallback returns None)
- Topic modeling (BERTopic if available; fallback to sklearn LDA)
- Clustering (KMeans on embeddings)
- Trend detection (per-restaurant review counts & moving averages)
- Emotion detection (transformers emotion model preferred; fallback to text2emotion)
- Toxicity detection (transformers toxic model preferred; fallback to profanity heuristic)
- All heavy models are optional and loaded in try/except (graceful failures).
"""

import ast
import re
import warnings
from typing import List, Optional, Union, Any
import pandas as pd
import numpy as np

# ------------------------
# Existing date utilities
# ------------------------

def fix_stringified_date_lists(df, column="dates"):
    """
    Converts stringified Python lists into real lists.
    Safe: uses ast.literal_eval only when the cell is a string starting with '['.
    """
    def _parse(x):
        if isinstance(x, str) and x.startswith("["):
            try:
                return ast.literal_eval(x)
            except Exception:
                warnings.warn(f"Could not parse date list: {x}")
                return x
        return x

    df[column] = df[column].apply(_parse)
    return df


def deduplicate_date_lists(df, column="dates"):
    """
    Removes duplicate date strings within each list while preserving order.
    """
    df[column] = df[column].apply(
        lambda lst: list(dict.fromkeys(lst)) if isinstance(lst, list) else lst
    )
    return df


def parse_date_list(date_list: Union[List[str], float]) -> Optional[List[pd.Timestamp]]:
    """
    Convert a list of date strings into a list of pandas Timestamp objects.
    Returns None for non-list inputs.
    """
    if isinstance(date_list, list):
        return pd.to_datetime(date_list, errors="coerce").tolist()
    return None


def add_date_features(df: pd.DataFrame, date_column: str = "dates") -> pd.DataFrame:
    """
    Given a DataFrame with a column containing lists of date strings,
    this function:
      - Parses each list into datetime objects (dates_parsed)
      - Extracts earliest and latest dates per row
      - Adds two new columns: 'earliest_date' and 'latest_date'
    """
    df["dates_parsed"] = df[date_column].apply(parse_date_list)

    df["earliest_date"] = df["dates_parsed"].apply(
        lambda lst: min(lst) if isinstance(lst, list) and len(lst) > 0 else pd.NaT
    )
    df["latest_date"] = df["dates_parsed"].apply(
        lambda lst: max(lst) if isinstance(lst, list) and len(lst) > 0 else pd.NaT
    )
    return df


def extract_latest_review(df):
    return df["latest_date"].max()


def extract_oldest_review(df):
    return df["earliest_date"].min()


def sort_by_latest(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    return df.sort_values("latest_date", ascending=ascending)


def sort_by_earliest(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return df.sort_values("earliest_date", ascending=ascending)


def review_count(df):
    return df["dates"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)


def add_review_count(df):
    df["review_count"] = review_count(df)
    return df


def add_review_period(df):
    df["review_period_days"] = (df["latest_date"] - df["earliest_date"]).dt.days
    return df


def explode_dates(df):
    df_exploded = df.explode("dates_parsed").reset_index(drop=True)
    df_exploded = df_exploded[df_exploded["dates_parsed"].notna()]
    df_exploded.rename(columns={"dates_parsed": "review_date"}, inplace=True)
    return df_exploded


# plotting helpers
import matplotlib.pyplot as plt

def plot_review_history(df, name_column="name", restaurant_name=None):
    df_ex = explode_dates(df)
    if restaurant_name:
        df_ex = df_ex[df_ex[name_column] == restaurant_name]
    plt.figure(figsize=(10, 4))
    plt.scatter(df_ex["review_date"], [1] * len(df_ex), alpha=0.5)
    plt.title(f"Review History for {restaurant_name}")
    plt.yticks([])
    plt.xlabel("Date")
    plt.show()


def plot_review_trend(df):
    df_ex = explode_dates(df)
    df_daily = df_ex.groupby("review_date").size()
    plt.figure(figsize=(12, 5))
    plt.plot(df_daily.index, df_daily.values)
    plt.title("Total Review Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Reviews per Day")
    plt.show()


# ------------------------
# Review cleaning & basic NLP
# ------------------------

def clean_single_review(text: str) -> str:
    """
    Basic cleaning for one review string (for display or NLP).
    - normalize escaped newlines
    - collapse newlines to spaces
    - remove HTML tags
    - remove TripAdvisor 'More' artifacts
    - normalize ellipses and whitespace
    """
    if not isinstance(text, str):
        return text
    # Normalize escaped newlines first (literal \n)
    text = text.replace("\\n", "\n")
    # Convert newlines to spaces (we produce NLP-ready single-paragraph text)
    text = text.replace("\n", " ")
    # Remove HTML tags (if any)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove TripAdvisor 'More' tokens and common artifacts
    text = re.sub(r"\\bMore\\b", "", text, flags=re.IGNORECASE)
    # Collapse repeated ellipses to single dot
    text = re.sub(r"\.{3,}", ".", text)
    # Remove weird repeated whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_review_list(review_list):
    """
    Clean a list of reviews: remove empties, dedupe, clean strings.
    """
    if not isinstance(review_list, list):
        return review_list
    cleaned = [clean_single_review(r) for r in review_list if isinstance(r, str)]
    # Filter out empty strings
    cleaned = [r for r in cleaned if r.strip()]
    # Deduplicate while preserving order
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned


def clean_all_reviews(df, column="reviews"):
    df[column] = df[column].apply(clean_review_list)
    return df


# ------------------------
# Fix repeated/ mirrored date lists (new)
# ------------------------

def fix_repeated_date_lists(df, column="dates"):
    """
    Robustly fix the repeated-half / mirrored duplication bug in date lists.

    Behaviour:
      - If the cell is a stringified list, it is parsed with ast.literal_eval.
      - If the list contains Timestamps or date strings, we normalize to pandas.Timestamp internally
      - If the list is even-length and first_half == second_half, keeps first_half only
      - Removes exact duplicates while preserving first occurrence order
      - Returns a list of ISO-like date strings (YYYY-MM-DD) sorted newest → oldest
    """
    def clean_single(dates_cell):
        # parse stringified lists
        if isinstance(dates_cell, str) and dates_cell.startswith("["):
            try:
                dates_cell = ast.literal_eval(dates_cell)
            except Exception:
                # fallback: keep original
                return dates_cell

        if not isinstance(dates_cell, list):
            return dates_cell

        # Convert all entries to timestamps (coerce errors to NaT)
        parsed = pd.to_datetime(dates_cell, errors="coerce")
        # Drop NaT
        parsed = [d for d in parsed if pd.notna(d)]

        if len(parsed) == 0:
            return []

        # If even-length and mirrored halves, trim to first half
        n = len(parsed)
        if n % 2 == 0:
            half = n // 2
            first_half = parsed[:half]
            second_half = parsed[half:]
            # Compare equality of timestamps
            if first_half == second_half:
                parsed = first_half

        # Deduplicate while preserving order (timestamps are hashable)
        seen = set()
        dedup = []
        for d in parsed:
            if d not in seen:
                dedup.append(d)
                seen.add(d)

        # Sort newest → oldest
        dedup.sort(reverse=True)

        # Convert back to strings in day-month-year style if needed or ISO
        # We'll return strings in 'YYYY-MM-DD' for consistency
        return [d.strftime('%Y-%m-%d') for d in dedup]

    df[column] = df[column].apply(clean_single)
    return df


# ------------------------
# Optional heavy NLP models (graceful fallbacks)
# ------------------------

# spaCy for tokenization & lemmatization
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            pass
except Exception:
    nlp = None
    warnings.warn("spaCy not available. Some normalization/sentence splitting will be limited.", UserWarning)

# NLTK stopwords
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()
    warnings.warn("NLTK stopwords not available; skipping stopword removal.", UserWarning)

# TextBlob for sentiment + language detection fallback
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None
    warnings.warn("TextBlob not available. Sentiment & language detection will fallback or be limited.", UserWarning)

# VADER sentiment fallback (NLTK)
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

# sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _embed_model = None
    warnings.warn("sentence-transformers not available. Embeddings functions will return None.", UserWarning)

# BERTopic for topic modeling (preferred)
try:
    from bertopic import BERTopic
    _bertopic_available = True
except Exception:
    _bertopic_available = False

# sklearn LDA fallback if BERTopic is not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
except Exception:
    TfidfVectorizer = None
    CountVectorizer = None
    LatentDirichletAllocation = None
    KMeans = None
    warnings.warn("sklearn not fully available. LDA/clustering fallbacks may be limited.", UserWarning)

# Emotion detection model (transformers) fallback to text2emotion
try:
    from transformers import pipeline
    try:
        _emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    except Exception:
        try:
            _emotion_pipe = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion")
        except Exception:
            _emotion_pipe = None
except Exception:
    _emotion_pipe = None

# text2emotion fallback
try:
    import text2emotion as t2e
except Exception:
    t2e = None

# Toxicity detection model (transformers) fallback to profanity list
try:
    _toxic_pipe = pipeline("text-classification", model="unitary/toxic-bert") if 'pipeline' in globals() else None
except Exception:
    _toxic_pipe = None

# Simple profanity list (fallback)
_PROFANITY_SET = {
    "shit", "fuck", "bitch", "asshole", "damn", "crap", "bollocks", "arse",
}


# ------------------------
# Language detection
# ------------------------

def detect_language(text: str) -> str:
    if not isinstance(text, str):
        return "unknown"
    if TextBlob:
        try:
            return str(TextBlob(text).detect_language())
        except Exception:
            return "unknown"
    return "unknown"


def add_review_language(df, col="reviews"):
    df[f"{col}_language"] = df[col].apply(
        lambda lst: [detect_language(r) for r in lst] if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Sentence splitting
# ------------------------

def split_into_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def add_sentence_splits(df, col="reviews"):
    df[f"{col}_sentences"] = df[col].apply(
        lambda lst: [split_into_sentences(r) for r in lst] if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Normalization (lemmatize + stopwords)
# ------------------------

def normalize_text(text: str, lemmatize=True, remove_stopwords=True) -> str:
    if not isinstance(text, str):
        return text
    if not nlp:
        tokens = re.findall(r"\w+", text.lower())
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)
    doc = nlp(text.lower())
    tokens = []
    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if remove_stopwords and tok.text in stop_words:
            continue
        tokens.append(tok.lemma_ if lemmatize else tok.text)
    return " ".join(tokens)


def normalize_review_list(df, col="reviews", lemmatize=True, remove_stopwords=True):
    df[f"{col}_normalized"] = df[col].apply(
        lambda lst: [normalize_text(r, lemmatize, remove_stopwords) for r in lst] if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Sentiment analysis
# ------------------------

def compute_sentiment_textblob(text: str) -> Optional[float]:
    if not isinstance(text, str) or not TextBlob:
        return None
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return None


def compute_sentiment_vader(text: str) -> Optional[float]:
    if not isinstance(text, str) or not _vader:
        return None
    try:
        return float(_vader.polarity_scores(text)["compound"])
    except Exception:
        return None


def compute_sentiment(text: str) -> Optional[float]:
    s = compute_sentiment_textblob(text)
    if s is not None:
        return s
    return compute_sentiment_vader(text)


def add_sentiment_scores(df, col="reviews"):
    df[f"{col}_sentiment"] = df[col].apply(
        lambda lst: [compute_sentiment(r) for r in lst] if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Embeddings
# ------------------------

def embed_text(text: str) -> Optional[np.ndarray]:
    if not isinstance(text, str):
        return None
    if _embed_model:
        try:
            vec = _embed_model.encode(text)
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None
    return None


def embed_reviews(df, col="reviews"):
    df[f"{col}_embeddings"] = df[col].apply(
        lambda lst: [embed_text(r) for r in lst] if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Topic modeling
# ------------------------

def topic_model_bertopic(docs: List[str], verbose=False):
    if not _bertopic_available:
        raise RuntimeError("BERTopic not available in this environment.")
    topic_model = BERTopic(verbose=verbose)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def topic_model_lda(docs: List[str], n_topics=10, max_features=1000, random_state=42):
    if TfidfVectorizer is None or LatentDirichletAllocation is None:
        raise RuntimeError("sklearn LDA not available.")
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    lda.fit(X)
    return lda, vec, X


def run_topic_modeling(df, col="reviews_normalized", method="auto", n_topics=10):
    all_docs = []
    index_map = []
    for i, row in df[col].items():
        if isinstance(row, list):
            for j, r in enumerate(row):
                if isinstance(r, str) and r.strip():
                    all_docs.append(r)
                    index_map.append((i, j))
    if len(all_docs) == 0:
        warnings.warn("No documents found for topic modeling.")
        return df

    chosen = method
    if method == "auto":
        chosen = "bertopic" if _bertopic_available else "lda"

    topic_assignments = [None] * len(all_docs)
    try:
        if chosen == "bertopic" and _bertopic_available:
            model, topics, probs = topic_model_bertopic(all_docs)
            topic_assignments = topics
        elif chosen == "lda":
            lda, vec, X = topic_model_lda(all_docs, n_topics=n_topics)
            doc_topic = lda.transform(X)
            topic_assignments = doc_topic.argmax(axis=1).tolist()
        else:
            raise RuntimeError("Unknown or unavailable topic modeling method.")
    except Exception as e:
        warnings.warn(f"Topic modeling failed: {e}")
        return df

    topic_col = f"{col}_topics"
    df[topic_col] = df[col].apply(lambda lst: [None] * len(lst) if isinstance(lst, list) else lst)
    for (row_idx, rev_idx), topic_id in zip(index_map, topic_assignments):
        try:
            if isinstance(df.at[row_idx, topic_col], list):
                df.at[row_idx, topic_col][rev_idx] = int(topic_id) if topic_id is not None else None
        except Exception:
            pass
    return df


# ------------------------
# Clustering (KMeans on embeddings)
# ------------------------

def cluster_reviews_by_embeddings(df, col="reviews", embeddings_col=None, n_clusters=8):
    emb_col = embeddings_col or f"{col}_embeddings"
    docs = []
    index_map = []
    for i, row in df[emb_col].items():
        if isinstance(row, list):
            for j, vec in enumerate(row):
                if isinstance(vec, np.ndarray):
                    docs.append(vec)
                    index_map.append((i, j))
    if len(docs) == 0:
        warnings.warn("No embeddings found for clustering.")
        return df

    X = np.stack(docs)
    if KMeans is None:
        warnings.warn("sklearn KMeans not available; skipping clustering.")
        return df

    kmeans = KMeans(n_clusters=min(n_clusters, len(X)), random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_col = f"{col}_cluster_ids"
    df[cluster_col] = df[col].apply(lambda lst: [None] * len(lst) if isinstance(lst, list) else lst)
    for (row_idx, rev_idx), lab in zip(index_map, labels):
        try:
            if isinstance(df.at[row_idx, cluster_col], list):
                df.at[row_idx, cluster_col][rev_idx] = int(lab)
        except Exception:
            pass
    return df


# ------------------------
# Trend detection per restaurant
# ------------------------

def compute_review_trend(df, name_column="restaurant", window_days=30):
    df_ex = explode_dates(df)
    if name_column not in df_ex.columns:
        raise KeyError(f"{name_column} not found in exploded dataframe.")
    daily = df_ex.groupby([name_column, "review_date"]).size().rename("count").reset_index()
    results = []
    for name, group in daily.groupby(name_column):
        grp = group.set_index("review_date").asfreq("D", fill_value=0)
        grp["rolling_mean"] = grp["count"].rolling(window=window_days, min_periods=1).mean()
        grp = grp.reset_index()
        grp[name_column] = name
        results.append(grp)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame(columns=[name_column, "review_date", "count", "rolling_mean"])


# ------------------------
# Emotion detection
# ------------------------

def detect_emotion_transformers(text: str):
    if _emotion_pipe is None:
        return None
    try:
        out = _emotion_pipe(text)
        if isinstance(out, list):
            try:
                return {d["label"]: float(d.get("score", 0.0)) for d in out}
            except Exception:
                return None
        elif isinstance(out, dict):
            return {out.get("label"): float(out.get("score", 0.0))}
    except Exception:
        return None
    return None

def detect_emotion_text2emotion(text: str):
    if t2e is None:
        return None
    try:
        return t2e.get_emotion(text)
    except Exception:
        return None

def add_emotion_scores(df, col="reviews"):
    df[f"{col}_emotion"] = df[col].apply(
        lambda lst: [detect_emotion_transformers(r) or detect_emotion_text2emotion(r) for r in lst]
        if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Toxicity detection
# ------------------------

def detect_toxicity_transformer(text: str):
    if _toxic_pipe is None:
        return None
    try:
        out = _toxic_pipe(text)
        if isinstance(out, list):
            try:
                return {d.get("label", str(i)): float(d.get("score", 0.0)) for i, d in enumerate(out)}
            except Exception:
                return None
        elif isinstance(out, dict):
            return {k: float(v) for k, v in out.items()}
    except Exception:
        return None
    return None

def detect_toxicity_profanity(text: str):
    if not isinstance(text, str):
        return {"profanity_flag": False, "count": 0}
    tokens = re.findall(r"\w+", text.lower())
    found = [t for t in tokens if t in _PROFANITY_SET]
    return {"profanity_flag": len(found) > 0, "count": len(found), "found": list(set(found))}

def add_toxicity_scores(df, col="reviews"):
    df[f"{col}_toxicity"] = df[col].apply(
        lambda lst: [detect_toxicity_transformer(r) or detect_toxicity_profanity(r) for r in lst]
        if isinstance(lst, list) else lst
    )
    return df


# ------------------------
# Master pipeline - extended
# ------------------------

def full_nlp_pipeline(
    df: pd.DataFrame,
    col="reviews",
    lemmatize=True,
    remove_stopwords=True,
    embed=True,
    topic_model=True,
    topic_method="auto",
    n_topics=10,
    clustering=False,
    n_clusters=8,
    detect_emotion_flag=True,
    detect_toxicity_flag=True,
):
    df = clean_all_reviews(df, col)
    df = add_sentence_splits(df, col)
    df = add_review_language(df, col)
    df = normalize_review_list(df, col, lemmatize=lemmatize, remove_stopwords=remove_stopwords)
    df = add_sentiment_scores(df, col)
    if embed:
        df = embed_reviews(df, col)
    if topic_model:
        src_col = f"{col}_normalized" if f"{col}_normalized" in df.columns else col
        try:
            df = run_topic_modeling(df, col=src_col, method=topic_method, n_topics=n_topics)
        except Exception as e:
            warnings.warn(f"Topic modeling skipped/failed: {e}")
    if clustering and embed:
        try:
            df = cluster_reviews_by_embeddings(df, col=col, n_clusters=n_clusters)
        except Exception as e:
            warnings.warn(f"Clustering skipped/failed: {e}")
    if detect_emotion_flag:
        try:
            df = add_emotion_scores(df, col=col)
        except Exception as e:
            warnings.warn(f"Emotion detection skipped/failed: {e}")
    if detect_toxicity_flag:
        try:
            df = add_toxicity_scores(df, col=col)
        except Exception as e:
            warnings.warn(f"Toxicity detection skipped/failed: {e}")
    return df


# ------------------------
# Utilities: small helpers users will likely call
# ------------------------

def explode_reviews(df, reviews_col="reviews"):
    df_copy = df[[c for c in df.columns if c != reviews_col]].copy()
    exploded = df[[reviews_col]].explode(reviews_col).reset_index().rename(columns={"index": "orig_row"})
    exploded = exploded[exploded[reviews_col].notna()]
    exploded = exploded.rename(columns={reviews_col: "review_text"})
    for cand in ["restaurant", "name", "place", "business_name"]:
        if cand in df.columns:
            exploded[cand] = df.loc[exploded["orig_row"], cand].values
            break
    return exploded


# ------------------------
# Small convenience pipeline for dates + reviews
# ------------------------

def preprocess_all(df, reviews_col="reviews", dates_col="dates"):
    """
    Full lightweight preprocessing that focuses on:
      - fixing stringified date lists
      - fixing mirrored/duplicated date lists
      - deduplicating dates
      - cleaning reviews
      - adding parsed date features and counts

    This is the recommended entry point for quick cleaning.
    """
    # 1) ensure stringified lists are parsed
    df = fix_stringified_date_lists(df, column=dates_col)
    # 2) fix the mirrored duplication bug in dates
    df = fix_repeated_date_lists(df, column=dates_col)
    # 3) deduplicate any remaining duplicate date strings
    df = deduplicate_date_lists(df, column=dates_col)
    # 4) clean reviews column (assumes reviews column is already a list OR stringified list)
    # parse stringified review lists if needed, then clean
    def parse_and_clean_reviews(cell):
        if isinstance(cell, str) and cell.startswith("["):
            try:
                cell = ast.literal_eval(cell)
            except Exception:
                return []
        return clean_review_list(cell) if isinstance(cell, list) else []

    df[reviews_col] = df[reviews_col].apply(parse_and_clean_reviews)

    # 5) date features
    df = add_date_features(df, date_column=dates_col)
    df = add_review_count(df)
    df = add_review_period(df)
    return df


# ------------------------
# End of file
