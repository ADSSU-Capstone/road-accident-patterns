import re
import os
import glob
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CONFIG = {
    "date_col": "DATE",
    "time_col": "TIME",
    "location_col": "LOCATION",
    "weather_col": "WEATHER",
    "road_col": "ROAD CONDITION",
    "vehicle_col": "VEHICLE TYPE",
    "cause_col": "CAUSE",
    "severity_col": "SEVERITY",
    "gender_col": "GENDER",
    "year_col": "_year",
    "target_name": "_severity_label",
    "high_risk_keywords": ["fatal", "major", "severe", "serious"],
    "low_risk_keywords": ["minor", "no injury", "none"],
    "random_state": 42,
    "default_clusters": 4,
    "default_depth": 5,
}

COLUMN_ALIASES = {
    "date": "DATE",
    "time": "TIME",
    "location": "LOCATION",
    "weather": "WEATHER",
    "roadcondition": "ROAD CONDITION",
    "raodcondition": "ROAD CONDITION",
    "vehicletype": "VEHICLE TYPE",
    "cause": "CAUSE",
    "severity": "SEVERITY",
    "gender": "GENDER",
}

def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [clean_text(c).upper().replace(" ", "") for c in frame.columns]
    rename = {}
    for col in frame.columns:
        key = col.lower()
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
    frame = frame.rename(columns=rename)
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
    return frame

def derive_year_from_text(*candidates):
    for text in candidates:
        if text is None:
            continue
        match = re.search(r"(20\d{2})", str(text))
        if match:
            return int(match.group(1))
    return np.nan

def load_local_data():
    files = glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.csv")
    if not files:
        return pd.DataFrame()
        
    frames = []
    for filepath in files:
        name = os.path.basename(filepath)
        if name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath, encoding="utf-8-sig", low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
            df["_source_file"] = clean_text(name)
            df["_year"] = derive_year_from_text(name)
            frames.append(normalize_columns(df))
            continue
            
        xls = pd.ExcelFile(filepath)
        for sheet in xls.sheet_names:
            try:
                sheet_df = pd.read_excel(xls, sheet_name=sheet, header=0)
            except Exception:
                continue
            sheet_df["_source_file"] = clean_text(name)
            sheet_year = derive_year_from_text(name, sheet)
            sheet_df["_year"] = sheet_year
            frames.append(normalize_columns(sheet_df))
            
    if not frames:
        return pd.DataFrame()
        
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    return combined

def standardize_severity(series, keywords_high, keywords_low):
    def classify(value):
        text = clean_text(value).lower()
        if any(k in text for k in keywords_high):
            return "High-Risk"
        if any(k in text for k in keywords_low):
            return "Low-Risk"
        return np.nan
    return series.apply(classify)

def parse_time_hour(series):
    def parse(value):
        text = clean_text(value)
        if not text:
            return np.nan
        match = re.match(r"^(\d{1,2}):(\d{2})", text)
        if match:
            return int(match.group(1))
        return np.nan
    return series.apply(parse)

def normalise_location(value):
    text = clean_text(value).upper()
    if not text:
        return ""
    text = re.sub(r"\bP\d+[A-Z]?\b", "", text)
    text = re.sub(r"\bC\d+\b", "", text)
    text = re.sub(r"\bCAMPO\s*\d+\b", "", text)
    text = re.sub(r"\bBARANGAY\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    return text

def prepare_features(frame, cfg):
    df = frame.copy()
    severity_raw = df.get(cfg["severity_col"], pd.Series(dtype=object))
    df[cfg["target_name"]] = standardize_severity(
        severity_raw, cfg["high_risk_keywords"], cfg["low_risk_keywords"]
    )
    df = df[df[cfg["target_name"]].notna()].copy()

    if cfg["date_col"] in df.columns:
        df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]], errors="coerce")
    else:
        df[cfg["date_col"]] = pd.NaT

    if cfg["year_col"] in df.columns:
        df[cfg["year_col"]] = pd.to_numeric(df[cfg["year_col"]], errors="coerce")
        df[cfg["year_col"]] = df[cfg["year_col"]].fillna(df[cfg["date_col"]].dt.year)
    else:
        df[cfg["year_col"]] = df[cfg["date_col"]].dt.year

    df["_month"] = df[cfg["date_col"]].dt.month
    df["_hour"] = parse_time_hour(df.get(cfg["time_col"], pd.Series([""] * len(df))))

    if cfg["location_col"] in df.columns:
        df["_loc_key"] = df[cfg["location_col"]].apply(normalise_location)
    else:
        df["_loc_key"] = ""

    return df

def build_model_frame(df, cfg):
    feature_cols = [
        cfg["weather_col"], cfg["road_col"], cfg["vehicle_col"],
        cfg["cause_col"], cfg["gender_col"], "_month", "_hour",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    encoded = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in ("_month", "_hour"):
            encoded[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            encoded[col] = df[col].astype(str).apply(clean_text)
            encoded[col] = LabelEncoder().fit_transform(encoded[col])
    target = (df[cfg["target_name"]] == "High-Risk").astype(int)
    keep = encoded.notna().all(axis=1)
    return encoded[keep], target[keep], feature_cols

def cluster_hotspots(df, cfg, n_clusters):
    location_freq = df["_loc_key"].value_counts()
    features = pd.DataFrame({
        "location_freq": df["_loc_key"].map(location_freq).fillna(0),
        "month": df["_month"].fillna(1),
        "hour": df["_hour"].fillna(12),
        "severity_score": df[cfg["target_name"]].map({"Low-Risk": 0, "High-Risk": 1}).fillna(0),
    })
    features = features.dropna()
    if len(features) < max(n_clusters * 2, 10):
        return None
    scaled = (features - features.mean()) / features.std(ddof=0).replace(0, 1)
    model = KMeans(n_clusters=n_clusters, random_state=cfg["random_state"], n_init=10)
    labels = model.fit_predict(scaled)
    score = silhouette_score(scaled, labels) if len(set(labels)) > 1 else float("nan")
    features = features.copy()
    features["cluster"] = labels
    features["location"] = df.loc[features.index, "_loc_key"].values
    return {"features": features, "silhouette": score, "model": model}

def cramers_v(table):
    chi2, _, _, _ = chi2_contingency(table)
    n = table.values.sum()
    if n == 0:
        return float("nan")
    r, c = table.shape
    return float(np.sqrt(chi2 / (n * max(r - 1, c - 1))))

def association_pairs(df, columns):
    rows = []
    for a, b in combinations(columns, 2):
        table = pd.crosstab(df[a].astype(str), df[b].astype(str))
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        try:
            v = cramers_v(table)
        except Exception:
            continue
        if np.isnan(v):
            continue
        rows.append({"variable_a": a, "variable_b": b, "cramers_v": round(v, 4)})
    if not rows:
        return pd.DataFrame(columns=["variable_a", "variable_b", "cramers_v"])
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)

def train_tree(features, target, depth, seed):
    if len(features) < 20 or target.nunique() < 2:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=seed, stratify=target
    )
    model = DecisionTreeClassifier(max_depth=depth, criterion="gini",
                                   class_weight="balanced", random_state=seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(model, features, target, cv=cv, scoring="accuracy")
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    importance = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    return {
        "model": model, "metrics": metrics, "cv": cv_scores,
        "confusion": cm, "report": report, "importance": importance,
        "n_train": len(x_train), "n_test": len(x_test),
    }

def bar_chart(series, title, color="#8B0000", top=None):
    data = series.dropna()
    if top:
        data = data.head(top)
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(data))))
    ax.barh(data.index.astype(str), data.values, color=color, edgecolor="black")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig

def line_chart(series, title, color="#1f4e79"):
    data = series.dropna()
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data.index.astype(str), data.values, marker="o", color=color)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

st.set_page_config(page_title="Road Accident Dashboard", page_icon="", layout="wide")

st.title("Road Accident Dashboard")
st.caption("Computed results from historical accident records")

raw = load_local_data()

if raw.empty:
    st.error("No dataset found. Please upload a dataset file (e.g., dataset.xlsx) to your GitHub repository alongside this app.py file.")
    st.stop()

prepared = prepare_features(raw, CONFIG)

available_years = sorted(prepared[CONFIG["year_col"]].dropna().astype(int).unique().tolist())
selected_years = st.sidebar.multiselect("Filter years", available_years, default=available_years)
if selected_years:
    prepared = prepared[prepared[CONFIG["year_col"]].isin(selected_years)].copy()

if prepared.empty:
    st.warning("No records remain after the year filter.")
    st.stop()

cluster_count = st.sidebar.slider("Number of clusters", 2, 8, CONFIG["default_clusters"])
tree_depth = st.sidebar.slider("Tree depth", 2, 12, CONFIG["default_depth"])

tabs = st.tabs([
    "Overview",
    "Results Across Time",
    "Variable Relationships",
    "Accident Hotspots",
    "Significant Factors",
    "Model Performance",
    "Decision Support",
])

with tabs[0]:
    total = len(prepared)
    high = int((prepared[CONFIG["target_name"]] == "High-Risk").sum())
    low = int((prepared[CONFIG["target_name"]] == "Low-Risk").sum())
    cols = st.columns(4)
    cols[0].metric("Records", f"{total:,}")
    cols[1].metric("High-Risk", f"{high:,}")
    cols[2].metric("Low-Risk", f"{low:,}")
    cols[3].metric("Share high-risk", f"{high / total:.1%}" if total else "-")

    st.subheader("Severity distribution")
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = prepared[CONFIG["target_name"]].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["#C0392B", "#27AE60"], startangle=90)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Records per year")
    year_counts = prepared[CONFIG["year_col"]].value_counts().sort_index()
    fig = bar_chart(year_counts, "Accidents per year", color="#34495E")
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Top cause categories")
    if CONFIG["cause_col"] in prepared.columns:
        cause_counts = prepared[CONFIG["cause_col"]].astype(str).apply(clean_text).value_counts().head(10)
        fig = bar_chart(cause_counts, "Top cause categories", color="#7F8C8D")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

with tabs[1]:
    st.subheader("Monthly trend")
    monthly = prepared.dropna(subset=["_month"]).groupby("_month").size()
    fig = line_chart(monthly, "Accidents per month")
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Hourly trend")
    hourly = prepared.dropna(subset=["_hour"]).groupby("_hour").size()
    fig = line_chart(hourly, "Accidents per hour", color="#8B0000")
    if fig:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Severity by year")
    if prepared[CONFIG["year_col"]].notna().any():
        by_year = pd.crosstab(prepared[CONFIG["year_col"]], prepared[CONFIG["target_name"]])
        st.dataframe(by_year, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        by_year.plot(kind="bar", ax=ax, color=["#27AE60", "#C0392B"], edgecolor="black")
        ax.set_ylabel("Records")
        ax.set_title("Severity by year")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with tabs[2]:
    st.subheader("Severity vs categorical variables")
    cat_cols = [c for c in [CONFIG["weather_col"], CONFIG["road_col"],
                            CONFIG["vehicle_col"], CONFIG["cause_col"],
                            CONFIG["gender_col"]] if c in prepared.columns]
    for col in cat_cols:
        table = pd.crosstab(prepared[col].astype(str).apply(clean_text),
                            prepared[CONFIG["target_name"]])
        if table.empty:
            continue
        try:
            v = cramers_v(table)
        except Exception:
            v = float("nan")
        st.markdown(f"**{col}** — Cramer's V = {v:.3f}" if not np.isnan(v) else f"**{col}**")
        st.dataframe(table, use_container_width=True)

    st.subheader("Top pairwise associations")
    assoc = association_pairs(prepared, cat_cols)
    if not assoc.empty:
        st.dataframe(assoc, use_container_width=True)
        top = assoc.iloc[0]
        st.write(
            f"Strongest association: {top['variable_a']} and {top['variable_b']} "
            f"(Cramer's V = {top['cramers_v']})."
        )

with tabs[3]:
    st.subheader("Cluster summary")
    result = cluster_hotspots(prepared, CONFIG, cluster_count)
    if result is None:
        st.warning("Not enough records to cluster with this configuration.")
    else:
        st.metric("Silhouette score", f"{result['silhouette']:.4f}")
        profile = result["features"].groupby("cluster").agg(
            records=("location", "size"),
            mean_location_freq=("location_freq", "mean"),
            mean_month=("month", "mean"),
            mean_hour=("hour", "mean"),
            high_risk_share=("severity_score", "mean"),
        ).round(3)
        st.dataframe(profile, use_container_width=True)

        st.subheader("Top locations per cluster")
        for cluster_id in sorted(result["features"]["cluster"].unique()):
            subset = result["features"][result["features"]["cluster"] == cluster_id]
            top_locations = subset["location"].value_counts().head(5)
            st.markdown(f"Cluster {cluster_id}")
            st.dataframe(top_locations.rename("Records").to_frame(), use_container_width=True)

        st.subheader("Cluster sizes")
        sizes = result["features"]["cluster"].value_counts().sort_index()
        fig = bar_chart(sizes, "Records per cluster", color="#16A085")
        if fig:
            st.pyplot(fig)
            plt.close(fig)

with tabs[4]:
    st.subheader("Decision tree feature importance")
    features, target, feature_cols = build_model_frame(prepared, CONFIG)
    if features.empty or target.nunique() < 2:
        st.warning("Not enough encoded records to fit a model.")
    else:
        tree = train_tree(features, target, tree_depth, CONFIG["random_state"])
        if tree is None:
            st.warning("Model could not be trained with the current configuration.")
        else:
            imp = tree["importance"]
            fig = bar_chart(imp, "Feature importance (Gini)", color="#8B0000")
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            st.dataframe(imp.rename("Importance").to_frame(), use_container_width=True)

            st.subheader("Severity share by top feature")
            top_feature = imp.index[0]
            if top_feature in prepared.columns:
                share = prepared.groupby(prepared[top_feature].astype(str).apply(clean_text))[CONFIG["target_name"]].apply(
                    lambda s: (s == "High-Risk").mean()
                ).sort_values(ascending=False).head(10)
                st.dataframe(share.rename("High-risk share").to_frame().round(3), use_container_width=True)

with tabs[5]:
    st.subheader("Model metrics")
    features, target, feature_cols = build_model_frame(prepared, CONFIG)
    if features.empty or target.nunique() < 2:
        st.warning("Not enough encoded records to evaluate the model.")
    else:
        tree = train_tree(features, target, tree_depth, CONFIG["random_state"])
        if tree is None:
            st.warning("Model could not be trained with the current configuration.")
        else:
            m = tree["metrics"]
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{m['accuracy']:.4f}")
            cols[1].metric("Precision", f"{m['precision']:.4f}")
            cols[2].metric("Recall", f"{m['recall']:.4f}")
            cols[3].metric("F1-Score", f"{m['f1']:.4f}")

            st.subheader("Confusion matrix")
            cm = tree["confusion"]
            labels = ["Low-Risk", "High-Risk"]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("5-fold cross-validation accuracy")
            cv = tree["cv"]
            st.write(f"Mean = {cv.mean():.4f}, Std = {cv.std():.4f}")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar([f"Fold {i + 1}" for i in range(len(cv))], cv, color="#1f4e79", edgecolor="black")
            ax.axhline(cv.mean(), color="green", linestyle="--")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with tabs[6]:
    st.subheader("Highest-risk combinations")
    if CONFIG["vehicle_col"] in prepared.columns and CONFIG["cause_col"] in prepared.columns:
        combo = (prepared.groupby([
            prepared[CONFIG["vehicle_col"]].astype(str).apply(clean_text),
            prepared[CONFIG["cause_col"]].astype(str).apply(clean_text),
        ])[CONFIG["target_name"]]
            .agg(records="size", high_risk=lambda s: (s == "High-Risk").sum()))
        combo["high_risk_share"] = (combo["high_risk"] / combo["records"]).round(3)
        combo = combo[combo["records"] >= 5].sort_values("high_risk_share", ascending=False).head(10)
        st.dataframe(combo, use_container_width=True)

    st.subheader("Peak months by high-risk share")
    monthly = (prepared.dropna(subset=["_month"])
               .groupby("_month")[CONFIG["target_name"]]
               .apply(lambda s: (s == "High-Risk").mean())
               .sort_values(ascending=False).head(6))
    st.dataframe(monthly.rename("High-risk share").to_frame().round(3), use_container_width=True)

    st.subheader("Top risky locations")
    loc = (prepared.groupby("_loc_key")[CONFIG["target_name"]]
           .agg(records="size", high_risk=lambda s: (s == "High-Risk").sum()))
    loc["high_risk_share"] = (loc["high_risk"] / loc["records"]).round(3)
    loc = loc[loc["records"] >= 5].sort_values("high_risk_share", ascending=False).head(10)
    st.dataframe(loc, use_container_width=True)
