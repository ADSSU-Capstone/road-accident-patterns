import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, silhouette_score)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Road Accident Patterns Dashboard - Bunawan",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #8B0000;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #8B0000 0%, #C0392B 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 2rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #8B0000 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# COLUMN NORMALIZATION HELPER
# ============================================================
def normalize_columns(frame):
    """Strip whitespace/BOM, unify variant spellings, drop duplicate columns."""
    frame = frame.copy()
    frame.columns = [str(c).strip().replace('\ufeff', '') for c in frame.columns]

    canonical_map = {
        'DATE': 'DATE',
        'TIME': 'TIME',
        'LOCATION': 'LOCATION',
        'WEATHER': 'WEATHER',
        'ROADCONDITION': 'ROAD CONDITION',
        'RAODCONDITION': 'ROAD CONDITION',
        'VEHICLETYPE': 'VEHICLE TYPE',
        'CAUSE': 'CAUSE',
        'SEVERITY': 'SEVERITY',
        'GENDER': 'GENDER',
        '_SOURCESHEET': '_source_sheet',
    }

    def canonical(name):
        key = name.upper().replace(' ', '').replace('_', '').replace('-', '')
        return canonical_map.get(key, name)

    frame.columns = [canonical(c) for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated(keep='first')]
    return frame


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load all sheets from the Excel file. Looks in the current folder (Streamlit Cloud) and /content/data (Colab)."""
    search_paths = ['.', '/content/data']
    files = []
    for path in search_paths:
        if os.path.isdir(path):
            files += glob.glob(os.path.join(path, '*.xlsx'))
            files += glob.glob(os.path.join(path, '*.xls'))
            files += glob.glob(os.path.join(path, '*.csv'))

    if not files:
        return None, None

    preferred = [f for f in files if 'dataset' in os.path.basename(f).lower()]
    target_file = preferred[0] if preferred else files[0]

    try:
        if target_file.lower().endswith('.csv'):
            try:
                df = pd.read_csv(target_file, encoding='utf-8-sig', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(target_file, encoding='latin-1', low_memory=False)
            df['_source_sheet'] = 'CSV'
            df = normalize_columns(df)
        else:
            xls = pd.ExcelFile(target_file)
            sheet_frames = []
            for sheet_name in xls.sheet_names:
                try:
                    sheet_df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                    sheet_df['_source_sheet'] = str(sheet_name)
                    sheet_df = normalize_columns(sheet_df)
                    sheet_frames.append(sheet_df)
                except Exception as e:
                    st.warning(f"Could not read sheet '{sheet_name}': {e}")

            if not sheet_frames:
                st.error(f"No readable sheets in {target_file}")
                return None, None

            df = pd.concat(sheet_frames, ignore_index=True, sort=False)
            df = normalize_columns(df)

    except Exception as e:
        st.error(f"Error reading {target_file}: {e}")
        return None, None

    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    return df, os.path.basename(target_file)

    preferred = [f for f in files if 'dataset' in os.path.basename(f).lower()]
    target_file = preferred[0] if preferred else files[0]

    try:
        if target_file.lower().endswith('.csv'):
            try:
                df = pd.read_csv(target_file, encoding='utf-8-sig', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(target_file, encoding='latin-1', low_memory=False)
            df['_source_sheet'] = 'CSV'
            df = normalize_columns(df)
        else:
            xls = pd.ExcelFile(target_file)
            sheet_frames = []
            for sheet_name in xls.sheet_names:
                try:
                    sheet_df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                    sheet_df['_source_sheet'] = str(sheet_name)
                    sheet_df = normalize_columns(sheet_df)
                    sheet_frames.append(sheet_df)
                except Exception as e:
                    st.warning(f"Could not read sheet '{sheet_name}': {e}")

            if not sheet_frames:
                st.error(f"No readable sheets in {target_file}")
                return None, None

            df = pd.concat(sheet_frames, ignore_index=True, sort=False)
            df = normalize_columns(df)

    except Exception as e:
        st.error(f"Error reading {target_file}: {e}")
        return None, None

    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    return df, os.path.basename(target_file)


@st.cache_data
def preprocess_data(df, feature_cols, target_col='_derived_severity'):
    df = df.copy()

    if target_col not in df.columns:
        if 'SEVERITY' in df.columns:
            df[target_col] = df['SEVERITY'].astype(str).str.strip().str.lower()
        else:
            return None, None, None

    valid_mask = df[target_col].notna() & (df[target_col] != 'nan') & (df[target_col] != '')
    df = df[valid_mask].copy()

    def map_severity(val):
        v = str(val).strip().lower()
        if any(k in v for k in ['fatal', 'major', 'severe', 'serious']):
            return 'High-Risk'
        if any(k in v for k in ['minor', 'no injury', 'low', 'none']):
            return 'Low-Risk'
        return 'Low-Risk'

    df['_severity_label'] = df[target_col].apply(map_severity)
    y = (df['_severity_label'] == 'High-Risk').astype(int)

    X = df[feature_cols].copy()

    for col in X.columns:
        numeric_series = pd.to_numeric(X[col], errors='coerce')
        if numeric_series.notna().sum() >= 0.5 * len(X):
            X[col] = numeric_series.fillna(numeric_series.median() if numeric_series.notna().any() else 0)
        else:
            X[col] = X[col].astype(str).fillna('Unknown')
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    mask = ~X.isna().any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y, df


def detect_feature_columns(df, max_features=15):
    features = []
    exclude = ['_source_sheet', 'DATE', 'TIME', 'SEVERITY', '_derived_severity',
               '_severity_label', 'LOCATION', 'CAUSE']

    for col in df.columns:
        if col in exclude:
            continue
        col_lower = col.lower()
        if any(k in col_lower for k in ['date', 'time', 'id', 'index']):
            continue
        if df[col].dtype == 'object':
            if df[col].nunique(dropna=True) > 60:
                continue
        features.append(col)

    return features[:max_features]


# ============================================================
# TRAINING
# ============================================================
def train_decision_tree(X, y, test_size=0.2, random_state=42, max_depth=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state,
                                criterion='gini', class_weight='balanced')
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)
    y_prob = dt.predict_proba(X_test_scaled)[:, 1]

    return {
        'model': dt, 'y_pred': y_pred, 'y_prob': y_prob,
        'y_test': y_test, 'X_test': X_test_scaled,
        'X_train': X_train_scaled, 'y_train': y_train
    }, scaler, X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    }


# ============================================================
# LOAD DATA
# ============================================================
df_raw, filename = load_data()

st.markdown('<div class="main-header">🚦 Road Accident Patterns Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Road Accidents Patterns in Bunawan, Agusan del Sur Using Historical Data<br>'
            '<i>K-Means Clustering & Decision Tree Algorithm</i></div>', unsafe_allow_html=True)

# ============================================================
# DEBUG PANEL
# ============================================================
with st.expander("🔍 Debug: Detected Data", expanded=False):
    if df_raw is None:
        st.error("❌ No dataset found. Please upload DATASETS.xlsx or a CSV file.")
    else:
        st.success(f"✅ Loaded file: `{filename}`")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{df_raw.shape[0]:,}")
        c2.metric("Columns", df_raw.shape[1])
        if '_source_sheet' in df_raw.columns:
            c3.metric("Sheets Combined", df_raw['_source_sheet'].nunique())
        else:
            c3.metric("Sheets", 1)

        st.write("**All detected columns:**")
        st.code(list(df_raw.columns))

        if '_source_sheet' in df_raw.columns:
            st.write("**Records per year/sheet:**")
            sheet_counts = df_raw['_source_sheet'].value_counts().sort_index()
            st.dataframe(sheet_counts.rename('Records').to_frame(), use_container_width=True)

        st.write("**First 5 rows:**")
        st.dataframe(df_raw.head(), use_container_width=True)

        st.write("**Data types & missing values:**")
        info_df = pd.DataFrame({
            'dtype': df_raw.dtypes.astype(str),
            'missing': df_raw.isna().sum(),
            'missing_%': (df_raw.isna().sum() / len(df_raw) * 100).round(2),
            'unique': df_raw.nunique()
        })
        st.dataframe(info_df, use_container_width=True)

if df_raw is None:
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Model Configuration")

default_features = detect_feature_columns(df_raw)
all_cols = [c for c in df_raw.columns if not c.startswith('_') and c not in ['DATE', 'TIME']]

feature_cols = st.sidebar.multiselect(
    "Select predictor variables:",
    options=all_cols,
    default=default_features[:6]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
max_depth = st.sidebar.slider("Decision Tree max depth", 2, 15, 5)
n_clusters = st.sidebar.slider("K-Means: number of clusters", 2, 10, 4)

if len(feature_cols) < 2:
    st.warning("⚠️ Please select at least 2 predictor variables from the sidebar.")
    st.stop()

target_col = '_derived_severity'
X, y, df_clean = preprocess_data(df_raw, feature_cols, target_col)

if X is None or len(X) < 10:
    st.error("❌ Not enough valid data after preprocessing. Check the Debug panel above.")
    st.stop()

results, scaler, X_train, X_test, y_train, y_test = train_decision_tree(
    X, y, test_size=test_size, random_state=int(random_state), max_depth=max_depth
)

metrics_table = pd.DataFrame({
    'Decision Tree': compute_metrics(results['y_test'], results['y_pred'], results['y_prob'])
}).T

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🎯 K-Means Hotspots", "🌳 Decision Tree",
    "📋 Cross-Validation", "💡 Recommendations"
])

# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Records</p><h2>{len(df_clean):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Predictors Used</p><h2>{len(feature_cols)}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>High-Risk</p><h2>{(y==1).sum():,}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>Low-Risk</p><h2>{(y==0).sum():,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🏆 Decision Tree Performance")
    st.dataframe(metrics_table.style.format("{:.4f}").highlight_max(axis=0, color='#c6efce'), use_container_width=True)

    st.subheader("📋 Dataset Sample")
    st.dataframe(df_clean.head(10), use_container_width=True)

# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Severity Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_clean['_severity_label'].value_counts()
        colors = ['#e74c3c', '#2ecc71']
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors[:len(counts)], startangle=90)
        ax.set_title("High-Risk vs Low-Risk Accidents")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Class Counts")
        st.dataframe(counts.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)

    st.markdown("---")
    st.subheader("Accidents by Year")

    if 'DATE' in df_raw.columns:
        df_dates = df_raw.copy()
        df_dates['DATE_PARSED'] = pd.to_datetime(df_dates['DATE'], errors='coerce')
        df_dates['YEAR'] = df_dates['DATE_PARSED'].dt.year
        year_counts = df_dates['YEAR'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        year_counts.plot(kind='bar', color='#8B0000', ax=ax, edgecolor='black')
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Accidents")
        ax.set_title("Road Accidents per Year in Bunawan")
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Feature Distributions")

    cat_cols = [c for c in feature_cols if X[c].nunique() <= 15][:6]
    if cat_cols:
        n = len(cat_cols)
        fig, axes = plt.subplots((n+2)//3, 3, figsize=(15, 4*((n+2)//3)))
        axes = axes.flatten() if n > 1 else [axes]
        for i, col in enumerate(cat_cols):
            vc = df_clean[col].astype(str).value_counts().head(8)
            axes[i].barh(vc.index.astype(str), vc.values, color='#C0392B', edgecolor='black')
            axes[i].set_title(f"{col} (Top 8)")
            axes[i].invert_yaxis()
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Correlation Heatmap (Encoded Features)")
    corr_data = X.copy()
    corr_data['Severity'] = y.values
    corr = corr_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ------------------------------------------------------------
# TAB 3: K-MEANS HOTSPOTS
# ------------------------------------------------------------
with tab3:
    st.header("🎯 K-Means Clustering — Accident Hotspots")

    df_km = df_raw.copy()

    if 'LOCATION' in df_km.columns:
        df_km['LOCATION_CLEAN'] = df_km['LOCATION'].astype(str).str.strip().str.upper()
        df_km['LOCATION_KEY'] = df_km['LOCATION_CLEAN'].str.extract(
            r'(P\d+[A-Z]?|BARANGAY\s+\w+|CAMPO\s+\d+|C\d+|BUNAWAN BROOK|LIBERTAD|'
            r'CONSUELO|SAN TEODORO|SAN ANDRES|MAMBALILI|IMELDA|NUEVA ERA|POBLACION|SAN MARCOS)',
            expand=False).fillna(df_km['LOCATION_CLEAN'])

    if 'LOCATION_KEY' in df_km.columns:
        loc_counts = df_km['LOCATION_KEY'].value_counts()
        df_km['LOCATION_FREQ'] = df_km['LOCATION_KEY'].map(loc_counts).fillna(0)
    else:
        df_km['LOCATION_FREQ'] = 1

    df_km['DATE_PARSED'] = pd.to_datetime(df_km['DATE'], errors='coerce')
    df_km['MONTH'] = df_km['DATE_PARSED'].dt.month.fillna(1)
    df_km['HOUR'] = pd.to_datetime(df_km['TIME'].astype(str), errors='coerce').dt.hour.fillna(12)

    def sev_num(v):
        v = str(v).strip().lower()
        if 'fatal' in v: return 3
        if 'major' in v: return 2
        if 'minor' in v: return 1
        return 0
    df_km['SEV_NUM'] = df_km['SEVERITY'].apply(sev_num) if 'SEVERITY' in df_km.columns else 0

    km_features = ['LOCATION_FREQ', 'MONTH', 'HOUR', 'SEV_NUM']
    km_data = df_km[km_features].dropna()
    km_data = km_data[~km_data.isin([np.inf, -np.inf]).any(axis=1)]

    if len(km_data) < n_clusters * 2:
        st.warning("Not enough data for K-Means with this cluster count. Reduce clusters.")
    else:
        scaler_km = StandardScaler()
        km_scaled = scaler_km.fit_transform(km_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
        labels = kmeans.fit_predict(km_scaled)

        sil = silhouette_score(km_scaled, labels)

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", n_clusters)
        c2.metric("Silhouette Score", f"{sil:.4f}")
        c3.metric("Records Clustered", f"{len(labels):,}")

        st.caption(f"**Interpretation:** Silhouette scores closer to 1.0 indicate well-separated clusters. "
                   f"Score of {sil:.4f} suggests "
                   f"{'strong' if sil > 0.5 else 'moderate' if sil > 0.25 else 'weak'} cluster structure.")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Sizes")
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_km = plt.cm.Set2(np.linspace(0, 1, n_clusters))
            ax.bar([f"C{i}" for i in cluster_counts.index], cluster_counts.values,
                   color=colors_km, edgecolor='black')
            ax.set_ylabel("Number of Accidents")
            ax.set_title("Accidents per Cluster")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Cluster Profile (Mean Values)")
            km_df = km_data.copy()
            km_df['CLUSTER'] = labels
            st.dataframe(km_df.groupby('CLUSTER').mean().round(3), use_container_width=True)

        st.markdown("---")
        st.subheader("Top Locations by Cluster (Hotspots)")
        if 'LOCATION_KEY' in df_km.columns:
            df_km_valid = df_km.loc[km_data.index].copy()
            df_km_valid['CLUSTER'] = labels
            for c in sorted(df_km_valid['CLUSTER'].unique()):
                top_locs = df_km_valid[df_km_valid['CLUSTER'] == c]['LOCATION_KEY'].value_counts().head(5)
                st.markdown(f"**Cluster {c} — Top 5 Locations:**")
                st.dataframe(top_locs.rename('Accidents').to_frame(), use_container_width=True)

# ------------------------------------------------------------
# TAB 4: DECISION TREE
# ------------------------------------------------------------
with tab4:
    st.header("🌳 Decision Tree — Severity Classification")

    m = compute_metrics(results['y_test'], results['y_pred'], results['y_prob'])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
    c2.metric("Precision", f"{m['Precision']:.4f}")
    c3.metric("Recall",    f"{m['Recall']:.4f}")
    c4.metric("F1-Score",  f"{m['F1-Score']:.4f}")
    c5.metric("ROC-AUC",   f"{m['ROC-AUC']:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Low-Risk', 'High-Risk'],
                    yticklabels=['Low-Risk', 'High-Risk'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("ROC Curve")
        if len(set(results['y_test'])) > 1:
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_prob'])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color='#8B0000', lw=2, label=f"AUC = {m['ROC-AUC']:.3f}")
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("ROC curve requires both classes in the test set.")

    st.subheader("Classification Report")
    rep = classification_report(results['y_test'], results['y_pred'],
                                target_names=['Low-Risk', 'High-Risk'],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Importance (Significant Factors)")
    importances = pd.Series(results['model'].feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(importances)*0.4)))
    importances.plot(kind='barh', color='#8B0000', ax=ax, edgecolor='black')
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Decision Tree Feature Importance")
    st.pyplot(fig)
    plt.close()

    st.caption("Higher importance → stronger contribution to predicting High-Risk severity.")

    st.markdown("---")
    with st.expander("🌲 View Decision Tree Structure"):
        try:
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(results['model'], feature_names=feature_cols,
                      class_names=['Low-Risk', 'High-Risk'],
                      filled=True, rounded=True, fontsize=9, ax=ax)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Could not render tree: {e}")

# ------------------------------------------------------------
# TAB 5: CROSS-VALIDATION
# ------------------------------------------------------------
with tab5:
    st.header("📋 Stratified K-Fold Cross-Validation")

    k = st.slider("Number of folds", 3, 10, 5)

    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    dt_cv = cross_val_score(
        DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced'),
        X_scaled, y, cv=cv, scoring='accuracy'
    )

    cv_df = pd.DataFrame({'Decision Tree': dt_cv})
    cv_df.index = [f"Fold {i+1}" for i in range(k)]
    cv_df.loc['Mean'] = cv_df.mean()
    cv_df.loc['Std']  = cv_df.iloc[:-2].std()

    st.dataframe(cv_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    cv_df.iloc[:-2].plot(kind='bar', ax=ax, color='#8B0000', edgecolor='black', legend=False)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{k}-Fold Cross-Validation Accuracy per Fold")
    ax.axhline(dt_cv.mean(), color='green', linestyle='--', label=f"Mean = {dt_cv.mean():.4f}")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"**Mean CV Accuracy:** {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")

# ------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Interpretation")

    best_acc = metrics_table['Accuracy'].max()
    st.success(f"### 🏆 Decision Tree Accuracy: **{best_acc:.4f}**")

    st.markdown("---")
    st.subheader("🔑 Top Contributing Factors")
    importances = pd.Series(results['model'].feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_factors = importances.head(5)
    for i, (factor, imp) in enumerate(top_factors.items(), 1):
        st.markdown(f"**{i}. {factor}** — importance = {imp:.4f}")

    st.markdown("---")
    st.subheader("🎯 Recommendations for Bunawan, Agusan del Sur")

    st.markdown(f"""
    Based on the analysis of **{len(df_clean):,}** recorded accidents (2021–2025)
    using **K-Means Clustering** and **Decision Tree** algorithms:

    #### 1. For the Local Government Unit (LGU) of Bunawan
    - **Prioritize hotspot clusters** identified in the K-Means tab for road
      infrastructure improvements (lighting, signage, rumble strips).
    - Deploy **traffic enforcement** during high-risk hours and months based
      on temporal patterns.
    - Allocate **emergency response resources** near high-density accident clusters.

    #### 2. For the PNP Traffic Enforcement Unit
    - Intensify **checkpoints and patrols** in barangays with the highest
      accident frequencies (see Cluster top-locations).
    - Focus on **motorcycle-related violations** (helmet laws, overspeeding,
      alcohol-influenced driving) given motorcycle dominance in the data.

    #### 3. For the MDRRMO
    - Use the **hotspot map** to preposition response teams.
    - Improve **data recording consistency** — many records show inconsistent
      date formats and duplicate entries.

    #### 4. For Drivers and Pedestrians
    - Exercise **extreme caution during rainy/stormy weather** and at night.
    - Motorcycle riders should always wear **helmets and reflective gear**.
    - Avoid **alcohol-influenced driving**, a recurring cause in the records.

    #### 5. For Future Researchers
    - Integrate **GIS coordinates** for true spatial clustering.
    - Apply **Random Forest, XGBoost, or Neural Networks** for comparison.
    - Include **traffic volume and road geometry** as features.
    - Address class imbalance with **SMOTE**.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")
    csv = metrics_table.to_csv().encode('utf-8')
    st.download_button("Download Metrics CSV", csv, "model_metrics.csv", "text/csv")

    preds = pd.DataFrame({
        'Actual': y_test.map({0: 'Low-Risk', 1: 'High-Risk'}),
        'Predicted': pd.Series(results['y_pred']).map({0: 'Low-Risk', 1: 'High-Risk'})
    })
    st.download_button("Download Predictions CSV",
                       preds.to_csv(index=False).encode('utf-8'),
                       "predictions.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("🎓 Capstone Dashboard • Road Accidents Patterns in Bunawan, Agusan del Sur "
           "Using Historical Data • Puno & Manuel • Agusan del Sur State University • 2026")
'''

