import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import re
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crime Hotspot Prediction Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem; font-weight: bold; color: #7B241C;
        text-align: center; padding: 1rem 0;
        border-bottom: 3px solid #7B241C;
    }
    .sub-header {
        font-size: 1.05rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #7B241C 0%, #C0392B 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 8px 8px 0 0;
        padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #7B241C !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
MUNICIPALITY_FILES = {
    'Trento': ['Trento-Dataset-2020-2025.xlsx', 'trento'],
    'Bunawan': ['Bunawan-Dataset-2020-2025.xlsx', 'bunawan'],
    'Rosario': ['Rosario-Dataset-2020-2025.xlsx', 'rosario'],
    'San Francisco': ['San-Francisco-Dataset-2020-2025.xlsx', 'san-francisco', 'sanfrancisco'],
}

COLUMN_ALIASES = {
    'barangay': 'barangay',
    'date committed': 'date_committed',
    'datecommitted': 'date_committed',
    'time committed': 'time_committed',
    'timecommitted': 'time_committed',
    'stages of felony': 'stage_of_felony',
    'stageoffelony': 'stage_of_felony',
    'offense': 'offense',
    'offense type': 'offense_type',
    'offensetype': 'offense_type',
    'victims status': 'victim_status',
    'victim': 'victim',
    'victims age': 'victim_age',
    'victims gender': 'victim_gender',
    'victims occupation': 'victim_occupation',
    'suspects status': 'suspect_status',
    'suspect': 'suspect',
    'suspects age': 'suspect_age',
    'suspects gender': 'suspect_gender',
    'suspects occupation': 'suspect_occupation',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'lat': 'latitude',
    'lng': 'longitude',
    'municipal': 'municipality',
    'province': 'province',
    'incident type': 'incident_type',
    'incidenttype': 'incident_type',
    'firearms caliber': 'firearms_caliber',
    'firearms kind': 'firearms_kind',
    'firearms make': 'firearms_make',
    'firearms stat': 'firearms_status',
    'facaliber': 'firearms_caliber',
    'fakind': 'firearms_kind',
    'famake': 'firearms_make',
    'fastatus': 'firearms_status',
    'drug involved': 'drug_involved',
    'druginvolved': 'drug_involved',
    'vehicle kind': 'vehicle_kind',
    'vehiclekind': 'vehicle_kind',
    'vehiclechassisno': 'vehicle_chassis_no',
    'vehicle chassis no': 'vehicle_chassis_no',
    'narrative': 'narrative',
}


# ============================================================
# DATA LOADING
# ============================================================
def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace('\ufeff', '').lower() for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


def _find_file(municipality, search_paths):
    aliases = MUNICIPALITY_FILES[municipality]
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        all_files = glob.glob(os.path.join(path, '*.xlsx')) + \
                    glob.glob(os.path.join(path, '*.xls'))
        for f in all_files:
            fname = os.path.basename(f).lower()
            for alias in aliases:
                if alias.lower() in fname:
                    return f
    return None


@st.cache_data(show_spinner=True)
def load_all_data():
    search_paths = ['.', '/content/data']
    all_frames = []
    load_report = []

    for municipality in MUNICIPALITY_FILES.keys():
        filepath = _find_file(municipality, search_paths)
        if filepath is None:
            load_report.append((municipality, 'NOT FOUND', 0))
            continue

        try:
            xls = pd.ExcelFile(filepath)
            sheets_loaded = 0
            rows_loaded = 0

            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                    if df.empty:
                        continue
                    df = _normalize_columns(df)

                    if 'barangay' not in df.columns:
                        continue

                    df['_source_municipality'] = municipality
                    df['_source_sheet'] = str(sheet_name)

                    year_match = None
                    m = re.search(r'(20\d{2})', str(sheet_name))
                    if m:
                        year_match = int(m.group(1))
                    df['_source_year'] = year_match

                    all_frames.append(df)
                    sheets_loaded += 1
                    rows_loaded += len(df)
                except Exception as e:
                    continue

            load_report.append((municipality, os.path.basename(filepath), rows_loaded))
        except Exception as e:
            load_report.append((municipality, f'ERROR: {e}', 0))
            continue

    if not all_frames:
        return None, load_report

    df_all = pd.concat(all_frames, ignore_index=True, sort=False)
    df_all = df_all.dropna(how='all').dropna(axis=1, how='all')
    return df_all, load_report


# ============================================================
# PREPROCESSING
# ============================================================
def _extract_hour(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val).strip()
        if ':' in s:
            return int(s.split(':')[0])
        return pd.to_datetime(s, errors='coerce').hour
    except Exception:
        return np.nan


def _extract_gender(val):
    if pd.isna(val):
        return None
    try:
        s = str(val)
        for token in s.replace('(', '').replace(')', '').split('/'):
            t = token.strip().lower()
            if t in ('male', 'female', 'm', 'f'):
                return 'Male' if t.startswith('m') else 'Female'
        return None
    except Exception:
        return None


def _extract_age(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val)
        for token in s.replace('(', '').replace(')', '').split('/'):
            t = token.strip()
            if t.isdigit() and 0 < int(t) < 120:
                return int(t)
        return np.nan
    except Exception:
        return np.nan


@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()

    df['date_committed'] = pd.to_datetime(df['date_committed'], errors='coerce')
    df['year'] = df['date_committed'].dt.year
    df['month'] = df['date_committed'].dt.month
    df['day'] = df['date_committed'].dt.day
    df['day_of_week'] = df['date_committed'].dt.dayofweek

    if '_source_year' in df.columns:
        df['year'] = df['year'].fillna(df['_source_year'])

    if 'time_committed' in df.columns:
        df['hour'] = df['time_committed'].apply(_extract_hour)
    else:
        df['hour'] = np.nan

    def part_of_day(h):
        if pd.isna(h):
            return 'Unknown'
        h = int(h)
        if 5 <= h < 12: return 'Morning'
        if 12 <= h < 17: return 'Afternoon'
        if 17 <= h < 21: return 'Evening'
        return 'Night'
    df['part_of_day'] = df['hour'].apply(part_of_day)

    df['barangay'] = df['barangay'].astype(str).str.strip().str.upper()

    def categorize_offense(o):
        if pd.isna(o):
            return 'Other'
        s = str(o).lower()
        if any(k in s for k in ['murder', 'homicide', 'parricide', 'physical injur', 'assault', 'threat', 'rape', 'lascivious', 'sexual', 'abuse', 'harass']):
            return 'Crimes Against Persons'
        if any(k in s for k in ['robbery', 'theft', 'carnapping', 'cattle rustling', 'estafa', 'swindling', 'arson', 'malicious mischief', 'trespass', 'damage to property']):
            return 'Crimes Against Property'
        if any(k in s for k in ['drugs', 'firearm', 'illegal logging', 'forestry', 'gambling', 'mining', 'fisheries', 'trafficking']):
            return 'Special Laws'
        if any(k in s for k in ['reckless imprudence', 'vehicular', 'traffic']):
            return 'Traffic/Reckless'
        return 'Other'
    df['offense_category'] = df['offense'].apply(categorize_offense)

    if 'victim' in df.columns:
        df['victim_gender'] = df['victim'].apply(_extract_gender)
        df['victim_age'] = df['victim'].apply(_extract_age)
    else:
        df['victim_gender'] = None
        df['victim_age'] = np.nan

    if 'suspect' in df.columns:
        df['suspect_gender'] = df['suspect'].apply(_extract_gender)
        df['suspect_age'] = df['suspect'].apply(_extract_age)
    else:
        df['suspect_gender'] = None
        df['suspect_age'] = np.nan

    def severity_label(row):
        v = str(row.get('victim', '')).lower()
        if 'killed' in v or 'deceased' in v or 'found dead' in v:
            return 'High'
        if 'hospitalized' in v or 'injured' in v or 'wounded' in v:
            return 'Medium'
        return 'Low'
    df['severity'] = df.apply(severity_label, axis=1)

    df = df.dropna(subset=['barangay'])
    df = df[df['barangay'] != 'NAN']

    return df


# ============================================================
# LOAD & PREPROCESS
# ============================================================
df_raw, load_report = load_all_data()

st.markdown('<div class="main-header">🚨 Crime Hotspot Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Crime Hotspot Prediction Using Data Mining Techniques<br>'
            '<i>Trento • Bunawan • Rosario • San Francisco, Agusan del Sur (2020–2025)</i></div>',
            unsafe_allow_html=True)

with st.expander("📥 Dataset Load Report", expanded=(df_raw is None)):
    report_df = pd.DataFrame(load_report, columns=['Municipality', 'File', 'Rows Loaded'])
    st.dataframe(report_df, use_container_width=True)

if df_raw is None:
    st.error("❌ No datasets found. Make sure the 4 Excel files are in the repo root.")
    st.stop()

df = preprocess(df_raw)

with st.expander("🔍 Data Summary", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Municipalities", df['_source_municipality'].nunique())
    c3.metric("Barangays", df['barangay'].nunique())
    c4.metric("Years Covered", f"{int(df['year'].min())}–{int(df['year'].max())}" if df['year'].notna().any() else "N/A")

    st.write("**Records per Municipality:**")
    st.dataframe(df['_source_municipality'].value_counts().rename('Records').to_frame(), use_container_width=True)

    st.write("**Records per Year:**")
    st.dataframe(df['year'].value_counts().sort_index().rename('Records').to_frame(), use_container_width=True)

    st.write("**Offense Category Distribution:**")
    st.dataframe(df['offense_category'].value_counts().rename('Count').to_frame(), use_container_width=True)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("⚙️ Analysis Controls")

municipality_filter = st.sidebar.multiselect(
    "Filter by Municipality:",
    options=sorted(df['_source_municipality'].unique()),
    default=sorted(df['_source_municipality'].unique())
)

year_min, year_max = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

k_clusters = st.sidebar.slider("K-Means: number of clusters", 2, 10, 4)
dbscan_eps = st.sidebar.slider("DBSCAN: epsilon", 0.1, 3.0, 0.5, 0.1)
dbscan_minpts = st.sidebar.slider("DBSCAN: min samples", 2, 20, 5)
min_support = st.sidebar.slider("Association: min support", 0.005, 0.20, 0.02, 0.005)
min_confidence = st.sidebar.slider("Association: min confidence", 0.1, 1.0, 0.4, 0.05)
arima_forecast_steps = st.sidebar.slider("ARIMA: forecast horizon (months)", 3, 24, 12)
arima_auto = st.sidebar.checkbox("ARIMA: auto-select order by AIC", value=True)
arima_manual_order = st.sidebar.text_input("ARIMA manual order (p,d,q) — only if auto is OFF", "1,1,1")

df_f = df[
    (df['_source_municipality'].isin(municipality_filter)) &
    (df['year'].between(year_range[0], year_range[1]))
].copy()

if len(df_f) < 20:
    st.warning("⚠️ Too few records after filtering. Adjust the sidebar filters.")
    st.stop()


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "📈 EDA",
    "🎯 Clustering (K-Means + DBSCAN)",
    "🔗 Association Rules",
    "📉 Time Series (ARIMA)",
    "💡 Recommendations"
])


# ------------------------------------------------------------
# TAB 1: OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("📊 Executive Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Incidents</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Municipalities</p><h2>{df_f["_source_municipality"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_f["barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>High-Severity</p><h2>{(df_f["severity"]=="High").sum():,}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 Incidence by Municipality")
    mun_counts = df_f['_source_municipality'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    mun_counts.plot(kind='bar', color='#7B241C', edgecolor='black', ax=ax)
    ax.set_ylabel("Number of Incidents")
    ax.set_title("Crime Incidents per Municipality")
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("📋 Offense Category Breakdown")
    off_counts = df_f['offense_category'].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60', '#2980b9', '#8e44ad', '#95a5a6']
        ax.pie(off_counts.values, labels=off_counts.index, autopct='%1.1f%%',
               colors=colors[:len(off_counts)], startangle=90)
        ax.set_title("Offense Category Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(off_counts.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)


# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.header("📈 Exploratory Data Analysis")

    st.subheader("🕐 Temporal Patterns")
    col1, col2 = st.columns(2)
    with col1:
        by_year = df_f.groupby('year').size()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_year.plot(kind='bar', color='#7B241C', edgecolor='black', ax=ax)
        ax.set_ylabel("Incidents")
        ax.set_title("Incidents per Year")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        by_month = df_f.groupby('month').size()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_month.plot(kind='bar', color='#C0392B', edgecolor='black', ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Incidents")
        ax.set_title("Incidents per Month (aggregated)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        by_hour = df_f['hour'].dropna().astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(by_hour.index, by_hour.values, marker='o', color='#7B241C')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Incidents")
        ax.set_title("Incidents by Hour")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        by_pod = df_f['part_of_day'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        by_pod.plot(kind='bar', color='#E67E22', edgecolor='black', ax=ax)
        ax.set_ylabel("Incidents")
        ax.set_title("Incidents by Part of Day")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("🏘️ Top 15 Barangays by Incident Count")
    top_brgy = df_f.groupby(['_source_municipality', 'barangay']).size() \
                   .sort_values(ascending=False).head(15)
    top_brgy.index = [f"{m} – {b}" for m, b in top_brgy.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    top_brgy.sort_values().plot(kind='barh', color='#7B241C', edgecolor='black', ax=ax)
    ax.set_xlabel("Incidents")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("⚠️ Severity Distribution")
    col1, col2 = st.columns([1, 1])
    with col1:
        sev_counts = df_f['severity'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_sev = {'High': '#c0392b', 'Medium': '#e67e22', 'Low': '#27ae60'}
        ax.pie(sev_counts.values, labels=sev_counts.index, autopct='%1.1f%%',
               colors=[colors_sev.get(x, '#95a5a6') for x in sev_counts.index], startangle=90)
        ax.set_title("Severity Distribution")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(sev_counts.rename('Count').to_frame().assign(
            Percentage=lambda d: (d['Count']/d['Count'].sum()*100).round(2)
        ), use_container_width=True)

    st.markdown("---")
    st.subheader("👥 Victim Demographics")
    col1, col2 = st.columns(2)
    with col1:
        vg = df_f['victim_gender'].dropna().value_counts()
        if len(vg) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            vg.plot(kind='bar', color='#3498db', edgecolor='black', ax=ax)
            ax.set_ylabel("Incidents")
            ax.set_title("Victim Gender")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    with col2:
        v_age = df_f['victim_age'].dropna()
        if len(v_age) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(v_age, bins=20, color='#e74c3c', edgecolor='black')
            ax.set_xlabel("Victim Age")
            ax.set_ylabel("Frequency")
            ax.set_title("Victim Age Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ------------------------------------------------------------
# TAB 3: CLUSTERING
# ------------------------------------------------------------
with tab3:
    st.header("🎯 Clustering Analysis — Crime Hotspot Identification")

    st.markdown("### Clustering at Barangay Level")
    st.caption("Each barangay is described by its crime profile. Clusters reveal similar-risk areas.")

    offense_pivot = df_f.pivot_table(
        index=['_source_municipality', 'barangay'],
        columns='offense_category',
        values='offense',
        aggfunc='count',
        fill_value=0
    )
    time_pivot = df_f.pivot_table(
        index=['_source_municipality', 'barangay'],
        columns='part_of_day',
        values='offense',
        aggfunc='count',
        fill_value=0
    )
    time_pivot.columns = [f"TOD_{c}" for c in time_pivot.columns]

    sev_pivot = df_f.pivot_table(
        index=['_source_municipality', 'barangay'],
        columns='severity',
        values='offense',
        aggfunc='count',
        fill_value=0
    )
    sev_pivot.columns = [f"SEV_{c}" for c in sev_pivot.columns]

    total = df_f.groupby(['_source_municipality', 'barangay']).size().rename('TOTAL')

    geo = df_f.groupby(['_source_municipality', 'barangay']).agg(
        lat=('latitude', 'mean'),
        lng=('longitude', 'mean')
    )

    features = pd.concat([offense_pivot, time_pivot, sev_pivot, total, geo], axis=1).fillna(0)
    features = features[features['TOTAL'] > 0]

    if len(features) < k_clusters * 2:
        st.warning("Not enough barangays after filtering for clustering.")
    else:
        X_clust = features.drop(columns=['lat', 'lng'], errors='ignore')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clust)

        st.markdown("---")
        st.subheader(f"🔵 K-Means Clustering (K = {k_clusters})")

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)
        sil_km = silhouette_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", k_clusters)
        c2.metric("Silhouette Score", f"{sil_km:.4f}")
        c3.metric("Barangays Clustered", len(features))

        st.caption(
            f"**Interpretation:** Silhouette closer to 1.0 = better-separated clusters. "
            f"Score of **{sil_km:.4f}** suggests "
            f"{'excellent' if sil_km>0.7 else 'acceptable' if sil_km>0.5 else 'weak' if sil_km>0.25 else 'poor'} "
            f"clustering quality."
        )

        features_km = features.copy()
        features_km['CLUSTER_KM'] = labels_km

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cluster Sizes**")
            fig, ax = plt.subplots(figsize=(6, 4))
            counts_km = pd.Series(labels_km).value_counts().sort_index()
            colors_km = plt.cm.Set2(np.linspace(0, 1, k_clusters))
            ax.bar([f"C{i}" for i in counts_km.index], counts_km.values,
                   color=colors_km, edgecolor='black')
            ax.set_ylabel("Number of Barangays")
            ax.set_title("Barangays per Cluster (K-Means)")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**PCA Visualization**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels_km, cmap='tab10',
                            s=60, alpha=0.8, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space (K-Means)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Cluster Profiles (mean feature values):**")
        profile = features_km.groupby('CLUSTER_KM').mean(numeric_only=True).round(2)
        st.dataframe(profile, use_container_width=True)

        st.markdown("**🚨 Top Barangays per K-Means Cluster (Hotspots):**")
        for c in sorted(features_km['CLUSTER_KM'].unique()):
            sub = features_km[features_km['CLUSTER_KM'] == c].sort_values('TOTAL', ascending=False).head(5)
            st.markdown(f"**Cluster {c} — Top 5 by Total Incidents**")
            display_df = sub[['TOTAL']].copy()
            display_df.index = [f"{m} – {b}" for m, b in display_df.index]
            st.dataframe(display_df, use_container_width=True)

        st.markdown("---")
        st.subheader(f"🟣 DBSCAN Clustering (ε = {dbscan_eps}, minPts = {dbscan_minpts})")

        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minpts)
        labels_db = dbscan.fit_predict(X_scaled)

        n_clusters_db = len(set(labels_db) - {-1})
        n_noise_db = (labels_db == -1).sum()

        sil_db = 0
        if n_clusters_db > 1:
            mask = labels_db != -1
            if mask.sum() > 1:
                try:
                    sil_db = silhouette_score(X_scaled[mask], labels_db[mask])
                except Exception:
                    sil_db = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters Found", n_clusters_db)
        c2.metric("Silhouette (non-noise)", f"{sil_db:.4f}" if sil_db else "N/A")
        c3.metric("Noise Points", n_noise_db)

        features_db = features.copy()
        features_db['CLUSTER_DB'] = labels_db

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            counts_db = pd.Series(labels_db).value_counts().sort_index()
            colors_db = plt.cm.Set3(np.linspace(0, 1, len(counts_db)))
            ax.bar([f"C{i}" if i != -1 else "Noise" for i in counts_db.index],
                   counts_db.values, color=colors_db, edgecolor='black')
            ax.set_ylabel("Number of Barangays")
            ax.set_title("DBSCAN Cluster Sizes")
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels_db, cmap='tab10',
                            s=60, alpha=0.8, edgecolors='black')
            plt.colorbar(sc, label='Cluster')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Barangays in PCA Space (DBSCAN)")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if n_noise_db > 0:
            st.markdown("**Noise Barangays (flagged by DBSCAN):**")
            noise_df = features_db[features_db['CLUSTER_DB'] == -1][['TOTAL']].sort_values('TOTAL', ascending=False)
            noise_df.index = [f"{m} – {b}" for m, b in noise_df.index]
            st.dataframe(noise_df, use_container_width=True)


# ------------------------------------------------------------
# TAB 4: ASSOCIATION RULES
# ------------------------------------------------------------
with tab4:
    st.header("🔗 Association Rule Mining")

    def season(m):
        if pd.isna(m): return 'Unknown'
        m = int(m)
        if m in (12, 1, 2): return 'Season_Dry'
        if m in (3, 4, 5): return 'Season_Hot'
        if m in (6, 7, 8): return 'Season_Wet'
        return 'Season_LateWet'

    transactions_df = pd.DataFrame({
        'MUNI': df_f['_source_municipality'].astype(str),
        'BRGY': df_f['barangay'].astype(str),
        'OFFENSE': df_f['offense_category'].astype(str),
        'TOD': df_f['part_of_day'].astype(str),
        'SEASON': df_f['month'].apply(season),
        'SEVERITY': df_f['severity'].astype(str),
    })

    transactions = transactions_df.values.tolist()
    transactions = [[str(x) for x in row if x and x != 'nan' and x != 'Unknown'] for row in transactions]
    transactions = [t for t in transactions if len(t) >= 2]

    if len(transactions) < 20:
        st.warning("Not enough transactions after filtering.")
    else:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_ary, columns=te.columns_)

        st.markdown("---")
        st.subheader("📌 Apriori Algorithm")

        try:
            frequent_ap = apriori(basket, min_support=min_support, use_colnames=True)
        except Exception as e:
            frequent_ap = pd.DataFrame()
            st.error(f"Apriori error: {e}")

        if frequent_ap.empty:
            st.warning(f"No frequent itemsets found with min support = {min_support}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent Itemsets", len(frequent_ap))
            c2.metric("Min Support", f"{min_support:.3f}")
            c3.metric("Min Confidence", f"{min_confidence:.2f}")

            st.markdown("**Top 10 Frequent Itemsets (by support):**")
            top_itemsets = frequent_ap.sort_values('support', ascending=False).head(10).copy()
            top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(x)))
            st.dataframe(top_itemsets, use_container_width=True)

            try:
                rules_ap = association_rules(frequent_ap, metric='confidence', min_threshold=min_confidence)
                rules_ap = rules_ap.sort_values('lift', ascending=False)
            except Exception as e:
                rules_ap = pd.DataFrame()

            if not rules_ap.empty:
                st.markdown(f"**Top 10 Apriori Rules (out of {len(rules_ap)}):**")
                display_rules = rules_ap.head(10).copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
                st.dataframe(
                    display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4),
                    use_container_width=True
                )

        st.markdown("---")
        st.subheader("📌 FP-Growth Algorithm")

        try:
            frequent_fp = fpgrowth(basket, min_support=min_support, use_colnames=True)
        except Exception as e:
            frequent_fp = pd.DataFrame()
            st.error(f"FP-Growth error: {e}")

        if frequent_fp.empty:
            st.warning(f"No frequent itemsets found with min support = {min_support}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Frequent Itemsets", len(frequent_fp))
            c2.metric("Min Support", f"{min_support:.3f}")
            c3.metric("Min Confidence", f"{min_confidence:.2f}")

            st.markdown("**Top 10 Frequent Itemsets (FP-Growth):**")
            top_fp = frequent_fp.sort_values('support', ascending=False).head(10).copy()
            top_fp['itemsets'] = top_fp['itemsets'].apply(lambda x: ', '.join(sorted(x)))
            st.dataframe(top_fp, use_container_width=True)

            try:
                rules_fp = association_rules(frequent_fp, metric='confidence', min_threshold=min_confidence)
                rules_fp = rules_fp.sort_values('lift', ascending=False)
            except Exception as e:
                rules_fp = pd.DataFrame()

            if not rules_fp.empty:
                st.markdown(f"**Top 10 FP-Growth Rules (out of {len(rules_fp)}):**")
                display_fp = rules_fp.head(10).copy()
                display_fp['antecedents'] = display_fp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                display_fp['consequents'] = display_fp['consequents'].apply(lambda x: ', '.join(sorted(x)))
                st.dataframe(
                    display_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4),
                    use_container_width=True
                )


# ------------------------------------------------------------
# TAB 5: TIME SERIES ARIMA (FULL CORRECT IMPLEMENTATION)
# ------------------------------------------------------------
with tab5:
    st.header("📉 Time Series Forecasting (ARIMA)")
    st.caption("Forecast future monthly incident counts based on historical trends.")

    ts_df = df_f.dropna(subset=['date_committed']).copy()
    ts_df['ym'] = ts_df['date_committed'].dt.to_period('M').dt.to_timestamp()
    monthly = ts_df.groupby('ym').size().rename('incidents').asfreq('MS').fillna(0)

    if len(monthly) < 18:
        st.warning(f"Not enough monthly data ({len(monthly)} months) to fit ARIMA. "
                   "Broaden the year range or municipality filter (need at least 18 months).")
    else:
        # ---------- Historical plot ----------
        st.markdown("### 📊 Historical Monthly Incidents")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly.index, monthly.values, marker='o', color='#7B241C')
        ax.set_ylabel("Incidents")
        ax.set_title("Monthly Incident Trend")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ---------- Stationarity test ----------
        st.markdown("---")
        st.markdown("### 🔬 Stationarity Test (Augmented Dickey-Fuller)")
        try:
            adf_result = adfuller(monthly.dropna(), autolag='AIC')
            adf_stat, adf_p = adf_result[0], adf_result[1]
            c1, c2, c3 = st.columns(3)
            c1.metric("ADF Statistic", f"{adf_stat:.4f}")
            c2.metric("p-value", f"{adf_p:.4f}")
            c3.metric("Conclusion", "Stationary" if adf_p < 0.05 else "Non-stationary")

            if adf_p < 0.05:
                st.success("✅ Series is stationary (p < 0.05). ARIMA with d=0 may work.")
                suggested_d = 0
            else:
                st.info("ℹ️ Series is non-stationary (p ≥ 0.05). Differencing (d=1) recommended.")
                suggested_d = 1
        except Exception as e:
            st.warning(f"ADF test failed: {e}")
            suggested_d = 1

        # ---------- ACF/PACF ----------
        st.markdown("---")
        st.markdown("### 📈 ACF and PACF Plots")
        st.caption("Used to identify p (AR) and q (MA) orders.")
        col1, col2 = st.columns(2)
        try:
            with col1:
                fig, ax = plt.subplots(figsize=(6, 3))
                plot_acf(monthly.dropna(), lags=min(20, len(monthly)-1), ax=ax)
                ax.set_title("ACF")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3))
                plot_pacf(monthly.dropna(), lags=min(20, len(monthly)-1), ax=ax, method='ywm')
                ax.set_title("PACF")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        except Exception as e:
            st.warning(f"Could not plot ACF/PACF: {e}")

        # ---------- Train/Test split ----------
        st.markdown("---")
        st.markdown("### 🎯 ARIMA Model Fit")

        train_size = int(len(monthly) * 0.8)
        train, test = monthly[:train_size], monthly[train_size:]

        # ---------- Auto-select best order ----------
        if arima_auto:
            with st.spinner("Searching for best ARIMA order by AIC (this may take 30–60 seconds)..."):
                best_aic = np.inf
                best_order = (1, suggested_d, 1)
                results = []
                for p in range(0, 4):
                    for d in range(0, 3):
                        for q in range(0, 4):
                            try:
                                m = ARIMA(train, order=(p, d, q))
                                fit = m.fit()
                                if fit.aic < best_aic:
                                    best_aic = fit.aic
                                    best_order = (p, d, q)
                                results.append({'order': (p, d, q), 'aic': fit.aic})
                            except Exception:
                                continue

            st.success(f"✅ Best ARIMA order selected: **ARIMA{best_order}** (AIC = {best_aic:.2f})")

            # Show top 5 candidate orders
            try:
                top_candidates = pd.DataFrame(results).sort_values('aic').head(5)
                top_candidates['order'] = top_candidates['order'].apply(lambda x: f"ARIMA{x}")
                st.markdown("**Top 5 candidate orders by AIC:**")
                st.dataframe(top_candidates.rename(columns={'order': 'Model', 'aic': 'AIC'}).round(2),
                             use_container_width=True)
            except Exception:
                pass
        else:
            try:
                parts = [int(x.strip()) for x in arima_manual_order.split(',')]
                best_order = tuple(parts[:3])
            except Exception:
                best_order = (1, suggested_d, 1)
                st.warning(f"Invalid manual order. Using default ARIMA{best_order}.")
            st.info(f"Using manual order: **ARIMA{best_order}**")

        # ---------- Fit final model ----------
        try:
            model = ARIMA(train, order=best_order)
            fitted = model.fit()

            with st.expander("📋 ARIMA Model Summary"):
                st.text(str(fitted.summary()))

            # ---------- Forecast on test ----------
            forecast_test = fitted.forecast(steps=len(test))
            forecast_test.index = test.index

            # ---------- Forecast into future ----------
            future_dates = pd.date_range(
                start=monthly.index[-1] + pd.DateOffset(months=1),
                periods=arima_forecast_steps, freq='MS'
            )
            # Refit on full data for best future forecast
            full_model = ARIMA(monthly, order=best_order).fit()
            future_forecast = full_model.forecast(steps=arima_forecast_steps)
            future_forecast.index = future_dates

            # ---------- Metrics ----------
            y_true = test.values
            y_pred = forecast_test.values
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)
            data_range = float(monthly.max() - monthly.min())
            rmse_pct = (rmse / data_range * 100) if data_range > 0 else 0

            st.markdown("### 📊 Evaluation Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{rmse:.2f}")
            c2.metric("MAE", f"{mae:.2f}")
            c3.metric("MAPE", f"{mape:.1f}%")
            c4.metric("RMSE / Range", f"{rmse_pct:.1f}%")

            if rmse_pct <= 10:
                st.success("✅ Excellent forecasting accuracy (RMSE ≤ 10% of range).")
            elif rmse_pct <= 20:
                st.info("ℹ️ Acceptable forecasting performance (RMSE ≤ 20% of range).")
            else:
                st.warning("⚠️ Poor forecasting accuracy (RMSE > 20% of range). Consider a different order.")

            # ---------- Plot ----------
            st.markdown("### 📉 Forecast vs Actual")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly.index, monthly.values, label='Actual', color='#7B241C',
                    marker='o', markersize=4)
            ax.plot(forecast_test.index, forecast_test.values,
                    label='ARIMA test forecast', color='#2980b9',
                    linestyle='--', marker='x')
            ax.plot(future_forecast.index, future_forecast.values,
                    label=f'Future forecast ({arima_forecast_steps}m)',
                    color='#27ae60', linestyle='--', marker='s')
            ax.axvline(x=test.index[0], color='gray', linestyle=':', alpha=0.7, label='Train/Test split')
            ax.set_ylabel("Incidents")
            ax.set_title(f"ARIMA{best_order} Forecast of Monthly Incidents")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ---------- Forecast table ----------
            st.markdown("### 📋 Forecast Values")
            forecast_table = pd.DataFrame({
                'Forecast Month': future_forecast.index.strftime('%Y-%m'),
                'Predicted Incidents': np.maximum(future_forecast.values, 0).round(2)
            })
            st.dataframe(forecast_table, use_container_width=True)

            st.download_button(
                "📥 Download Forecast CSV",
                forecast_table.to_csv(index=False).encode('utf-8'),
                "arima_forecast.csv", "text/csv"
            )

        except Exception as e:
            st.error(f"ARIMA fitting failed: {e}")
            st.caption("Try a wider year range, fewer municipalities, or uncheck 'auto-select order' "
                       "and try a simpler order like (1,1,1) or (0,1,1).")

        # ---------- Rolling Window Cross-Validation ----------
        st.markdown("---")
        st.markdown("### 🔄 Rolling Window Cross-Validation")
        st.caption("Walk-forward validation: train on expanding window, predict next month, repeat.")

        try:
            cv_rmse_list = []
            cv_mae_list = []
            cv_preds = []
            cv_actuals = []

            min_train = max(12, int(len(monthly) * 0.5))
            horizon = 1  # predict next month

            progress = st.progress(0)
            total_folds = len(monthly) - min_train - horizon + 1
            fold_count = 0

            for i in range(min_train, len(monthly) - horizon + 1):
                train_cv = monthly[:i]
                test_cv = monthly[i:i + horizon]

                try:
                    m_cv = ARIMA(train_cv, order=best_order)
                    fit_cv = m_cv.fit()
                    pred_cv = fit_cv.forecast(steps=horizon)

                    cv_rmse_list.append(np.sqrt(mean_squared_error(test_cv.values, pred_cv.values)))
                    cv_mae_list.append(mean_absolute_error(test_cv.values, pred_cv.values))
                    cv_preds.append(pred_cv.values[0])
                    cv_actuals.append(test_cv.values[0])
                except Exception:
                    continue

                fold_count += 1
                if total_folds > 0:
                    progress.progress(min(fold_count / total_folds, 1.0))

            progress.empty()

            if len(cv_rmse_list) > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("Folds Evaluated", len(cv_rmse_list))
                c2.metric("Mean CV RMSE", f"{np.mean(cv_rmse_list):.2f}")
                c3.metric("Mean CV MAE", f"{np.mean(cv_mae_list):.2f}")

                cv_range = monthly.max() - monthly.min()
                cv_rmse_pct = (np.mean(cv_rmse_list) / cv_range * 100) if cv_range > 0 else 0
                st.info(f"Mean CV RMSE is **{cv_rmse_pct:.1f}%** of the data range. "
                        f"Chapter 3 threshold: ≤20% acceptable, ≤10% excellent.")

                # Plot rolling predictions vs actuals
                cv_index = monthly.index[min_train:min_train + len(cv_preds)]
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(cv_index, cv_actuals, label='Actual', color='#7B241C', marker='o', markersize=3)
                ax.plot(cv_index, cv_preds, label='ARIMA (rolling 1-step forecast)',
                        color='#2980b9', linestyle='--', marker='x', markersize=3)
                ax.set_ylabel("Incidents")
                ax.set_title("Rolling Window Cross-Validation Predictions")
                ax.legend()
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Not enough data to perform rolling cross-validation.")
        except Exception as e:
            st.warning(f"Rolling CV failed: {e}")


# ------------------------------------------------------------
# TAB 6: RECOMMENDATIONS
# ------------------------------------------------------------
with tab6:
    st.header("💡 Recommendations & Interpretation")

    st.markdown(f"""
    ### 📊 Analysis Summary

    Based on **{len(df_f):,}** crime incidents from **{df_f['_source_municipality'].nunique()}**
    municipalities (**{', '.join(sorted(df_f['_source_municipality'].unique()))}**) covering
    **{year_range[0]}–{year_range[1]}**, using data mining techniques (K-Means, DBSCAN,
    Apriori, FP-Growth, and ARIMA):

    #### 1. For Law Enforcement Agencies (PNP)
    - **Focus patrols on high-density clusters** identified in the Clustering tab.
      Clusters with the highest mean total incidents are your priority barangays.
    - **Align patrol schedules** with temporal patterns: most incidents happen during
      **{df_f['part_of_day'].value_counts().idxmax()}** hours.
    - **Use association rules** (Apriori / FP-Growth) to anticipate co-occurring
      conditions — e.g., specific offense types during specific seasons.

    #### 2. For Local Government Units (LGUs)
    - **Allocate resources** to barangays that appear at the top of every cluster's
      ranking — these are persistent hotspots.
    - **Use the ARIMA forecast** to plan budgets for patrol vehicles, CCTV, and
      street lighting in upcoming months.
    - **Align with PPAN / peace-and-order plans** of Agusan del Sur province.

    #### 3. For Community Residents
    - **Be extra cautious during evening/night hours** and in hotspot barangays.
    - **Report suspicious activities** to your barangay tanod or nearest PNP station.

    #### 4. For Future Researchers
    - **Add LSTM** for deep-learning time-series forecasting (compare against ARIMA).
    - **Integrate GIS coordinates** for true spatial analysis (Moran's I, Getis-Ord Gi*).
    - **Address class imbalance** with SMOTE for classification tasks.
    - **Include socio-economic variables** (population density, poverty incidence, etc.).
    - **Automate data ingestion** from PNP records for real-time hotspot updates.

    #### 5. Limitations
    - Only **reported** crimes are included — the "dark figure of crime" is not captured.
    - Some records lack latitude/longitude; geographic clustering uses barangay centroids.
    - Column names and formats vary slightly across municipalities; normalization is
      handled but may lose edge-case values.
    - ARIMA assumes linear relationships; non-linear patterns (e.g., pandemic effects)
      may not be fully captured.
    """)

    st.markdown("---")
    st.subheader("📥 Download Results")
    try:
        metrics_summary = pd.DataFrame({
            'Metric': ['Total Records', 'Municipalities', 'Barangays'],
            'Value': [
                len(df_f),
                df_f['_source_municipality'].nunique(),
                df_f['barangay'].nunique()
            ]
        })
        st.download_button(
            "📊 Download Summary CSV",
            metrics_summary.to_csv(index=False).encode('utf-8'),
            "crime_hotspot_summary.csv", "text/csv"
        )
    except Exception:
        pass


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🎓 Capstone Dashboard • Crime Hotspot Prediction Using Data Mining Techniques in "
    "Trento, Bunawan, Rosario, and San Francisco, Agusan del Sur • "
    "Casing, R.V. & Verano, S.I. • Agusan del Sur State University • 2026"
)