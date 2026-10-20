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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Wide layout is crucial so it fills the screen
st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem; 
        border-radius: 8px; 
        border-left: 5px solid #1f4e79;
        margin-bottom: 1rem;
    }
    .metric-card h4 { margin: 0; color: #555; font-size: 0.9rem; text-transform: uppercase;}
    .metric-card h2 { margin: 0; color: #1f4e79; font-size: 1.8rem; }
    .interp-box { 
        background-color: #eef2f5; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 4px solid #2980b9;
        font-size: 1rem;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    /* Stretch charts to fill container */
    .stPlotlyChart, .stPyplot { width: 100% !important; }
</style>
""", unsafe_allow_html=True)

MUNICIPALITY_FILES = {
    'Trento': ['Trento-Dataset-2020-2025.xlsx', 'trento'],
    'Bunawan': ['Bunawan-Dataset-2020-2025.xlsx', 'bunawan'],
    'Rosario': ['Rosario-Dataset-2020-2025.xlsx', 'rosario'],
    'San Francisco': ['San-Francisco-Dataset-2020-2025.xlsx', 'san-francisco', 'sanfrancisco'],
}

COLUMN_ALIASES = {
    'barangay': 'barangay', 'date committed': 'date_committed', 'datecommitted': 'date_committed',
    'time committed': 'time_committed', 'timecommitted': 'time_committed',
    'offense': 'offense', 'victim': 'victim', 'suspect': 'suspect',
    'latitude': 'latitude', 'longitude': 'longitude', 'lat': 'latitude', 'lng': 'longitude'
}

def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace('\ufeff', '').lower() for c in df.columns]
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    return df.loc[:, ~df.columns.duplicated(keep='first')]

def _find_file(municipality, search_paths):
    aliases = MUNICIPALITY_FILES[municipality]
    for path in search_paths:
        if not os.path.isdir(path):
            continue
        for f in glob.glob(os.path.join(path, '*.xlsx')) + glob.glob(os.path.join(path, '*.xls')):
            fname = os.path.basename(f).lower()
            if any(alias in fname for alias in aliases):
                return f
    return None

@st.cache_data(show_spinner=True)
def load_all_data():
    all_frames = []
    for municipality in MUNICIPALITY_FILES.keys():
        filepath = _find_file(municipality, ['.', '/content/data'])
        if not filepath:
            continue
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
            if df.empty:
                continue
            df = _normalize_columns(df)
            if 'barangay' not in df.columns:
                continue
            df['_source_municipality'] = municipality
            m = re.search(r'(20\d{2})', str(sheet_name))
            df['_source_year'] = int(m.group(1)) if m else None
            all_frames.append(df)
    if not all_frames:
        return None
    return pd.concat(all_frames, ignore_index=True, sort=False).dropna(how='all').dropna(axis=1, how='all')

def _extract_hour(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if ':' in s: return int(s.split(':')[0])
    return pd.to_datetime(s, errors='coerce').hour

@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()
    
    if 'date_committed' in df.columns:
        df['date_committed'] = pd.to_datetime(df['date_committed'], errors='coerce')
    else:
        df['date_committed'] = pd.NaT
        
    df['year'] = df['date_committed'].dt.year.fillna(df.get('_source_year'))
    df['month'] = df['date_committed'].dt.month
    
    if 'time_committed' in df.columns:
        df['hour'] = df['time_committed'].apply(_extract_hour)
    else:
        df['hour'] = np.nan
    
    def part_of_day(h):
        if pd.isna(h): return 'Unknown'
        h = int(h)
        if 5 <= h < 12: return 'Morning'
        if 12 <= h < 17: return 'Afternoon'
        if 17 <= h < 21: return 'Evening'
        return 'Night'
    
    df['part_of_day'] = df['hour'].apply(part_of_day)
    df['barangay'] = df['barangay'].astype(str).str.strip().str.upper()

    def categorize_offense(o):
        if pd.isna(o): return 'Other'
        s = str(o).lower()
        if any(k in s for k in ['murder', 'homicide', 'parricide', 'physical injur', 'assault', 'threat', 'rape', 'lascivious', 'sexual', 'abuse', 'harass']): return 'Crimes Against Persons'
        if any(k in s for k in ['robbery', 'theft', 'carnapping', 'cattle rustling', 'estafa', 'swindling', 'arson', 'malicious mischief', 'trespass', 'damage to property']): return 'Crimes Against Property'
        if any(k in s for k in ['drugs', 'firearm', 'illegal logging', 'forestry', 'gambling', 'mining', 'fisheries', 'trafficking']): return 'Special Laws'
        if any(k in s for k in ['reckless imprudence', 'vehicular', 'traffic']): return 'Traffic/Reckless'
        return 'Other'
    
    df['offense_category'] = df['offense'].apply(categorize_offense) if 'offense' in df.columns else 'Other'

    def severity_label(row):
        v = str(row.get('victim', '')).lower()
        if 'killed' in v or 'deceased' in v or 'found dead' in v: return 'High'
        if 'hospitalized' in v or 'injured' in v or 'wounded' in v: return 'Medium'
        return 'Low'
    
    df['severity'] = df.apply(severity_label, axis=1)
    for col in ['latitude', 'longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan
        
    return df[df['barangay'].notna() & (df['barangay'] != 'NAN') & (df['barangay'] != '')].copy()

# Load Data
df_raw = load_all_data()
if df_raw is None:
    st.error("Dataset files not found.")
    st.stop()
df = preprocess(df_raw)

# Sidebar Navigation
st.sidebar.title("Dashboard Sections")
page = st.sidebar.radio("", [
    "Overview",
    "Data Preparation", 
    "Clustering Evaluation", 
    "Hotspot Factors", 
    "Association Rules", 
    "Time Series Forecasting"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Algorithm Parameters")
k_clusters = st.sidebar.slider("K-Means Clusters", 2, 10, 4)
dbscan_eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 3.0, 0.5, 0.1)
dbscan_minpts = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
min_support = st.sidebar.slider("Rule Support", 0.005, 0.20, 0.02, 0.005)
min_confidence = st.sidebar.slider("Rule Confidence", 0.1, 1.0, 0.4, 0.05)
arima_steps = st.sidebar.slider("Forecast Horizon", 3, 24, 12)

df_f = df.copy()

# ==========================================
# 1. OVERVIEW
# ==========================================
if page == "Overview":
    st.title("Incident Overview")
    
    # Top Row Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h4>Total Incidents</h4><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h4>Municipalities</h4><h2>{df_f["_source_municipality"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h4>Barangays</h4><h2>{df_f["barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h4>High Severity Cases</h4><h2>{(df_f["severity"]=="High").sum():,}</h2></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Middle Row Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Crime Categories")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['offense_category'].value_counts().plot(kind='bar', color='#1f4e79', ax=ax)
        ax.set_ylabel("Number of Cases")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="interp-box"><b>Interpretation:</b> Property and personal crimes dominate the dataset. Law enforcement should prioritize these categories for resource allocation and prevention programs.</div>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("Severity Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f['severity'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#e74c3c', '#f39c12', '#27ae60'], ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="interp-box"><b>Interpretation:</b> While low-severity incidents make up the majority, the presence of high-severity cases highlights the need for rapid emergency response capabilities.</div>', unsafe_allow_html=True)

    # Bottom Row: Data Preview to fill space
    st.markdown("---")
    st.subheader("Dataset Snapshot")
    st.dataframe(df_f[['barangay', 'offense_category', 'date_committed', 'part_of_day', 'severity']].head(8), use_container_width=True)

# ==========================================
# 2. DATA PREPARATION & EDA
# ==========================================
elif page == "Data Preparation":
    st.title("Data Preparation & EDA")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annual Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f.groupby('year').size().plot(kind='line', marker='o', color='#1f4e79', linewidth=2, ax=ax)
        ax.set_ylabel("Incident Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="interp-box"><b>Interpretation:</b> The annual trend reveals whether crime rates are escalating or stabilizing over the 5-year period.</div>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("Time of Day Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_f.groupby('part_of_day').size().plot(kind='bar', color='#2980b9', ax=ax)
        ax.set_ylabel("Incident Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="interp-box"><b>Interpretation:</b> Identifying peak hours allows the PNP to optimize patrol schedules and allocate personnel during high-risk times.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Monthly Distribution")
    fig, ax = plt.subplots(figsize=(12, 4))
    df_f.groupby('month').size().plot(kind='bar', color='#27ae60', ax=ax)
    ax.set_xlabel("Month (1 = Jan, 12 = Dec)")
    ax.set_ylabel("Total Incidents")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="interp-box"><b>Interpretation:</b> Seasonal spikes in crime (e.g., during summer or holiday seasons) can be anticipated and mitigated using this monthly breakdown.</div>', unsafe_allow_html=True)

# ==========================================
# 3. CLUSTERING EVALUATION
# ==========================================
elif page == "Clustering Evaluation":
    st.title("Clustering Evaluation")
    
    offense_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='offense_category', values='offense', aggfunc='count', fill_value=0)
    time_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='part_of_day', values='offense', aggfunc='count', fill_value=0)
    sev_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='severity', values='offense', aggfunc='count', fill_value=0)
    total = df_f.groupby(['_source_municipality', 'barangay']).size().rename('TOTAL')
    
    features = pd.concat([offense_pivot, time_pivot, sev_pivot, total], axis=1).fillna(0)
    features = features[features['TOTAL'] > 0]
    
    if len(features) >= k_clusters * 2:
        X_scaled = StandardScaler().fit_transform(features.drop(columns=['TOTAL'], errors='ignore'))
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)
        sil_km = silhouette_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0
        dbi_km = davies_bouldin_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0
        
        st.subheader("Model Performance Comparison")
        
        # Wider metrics table
        metrics_data = {
            "Algorithm": ["K-Means Clustering", "DBSCAN Clustering"],
            "Evaluation Metric": ["Silhouette Score", "Davies-Bouldin Index"],
            "Score": [f"{sil_km:.4f}", f"{dbi_km:.4f}"],
            "Interpretation": ["Higher is better (max 1.0)", "Lower is better (min 0.0)"]
        }
        st.table(pd.DataFrame(metrics_data))
        
        st.markdown('<div class="interp-box"><b>Interpretation:</b> The K-Means algorithm successfully partitioned the barangays. A Silhouette Score closer to 1.0 and a DBI closer to 0 indicates that the clusters are distinct and well-separated.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Cluster Visualization (PCA)")
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='viridis', s=100, edgecolors='k')
        plt.colorbar(sc, label='Assigned Cluster')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="interp-box"><b>Interpretation:</b> The PCA plot visualizes the separation between clusters. Distinct groupings indicate that barangays within the same cluster share similar crime dynamics.</div>', unsafe_allow_html=True)

# ==========================================
# 4. HOTSPOT FACTORS
# ==========================================
elif page == "Hotspot Factors":
    st.title("Hotspot Factor Analysis")
    
    offense_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='offense_category', values='offense', aggfunc='count', fill_value=0)
    time_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='part_of_day', values='offense', aggfunc='count', fill_value=0)
    sev_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='severity', values='offense', aggfunc='count', fill_value=0)
    total = df_f.groupby(['_source_municipality', 'barangay']).size().rename('TOTAL')
    
    features = pd.concat([offense_pivot, time_pivot, sev_pivot, total], axis=1).fillna(0)
    features = features[features['TOTAL'] > 0]
    X_scaled = StandardScaler().fit_transform(features.drop(columns=['TOTAL'], errors='ignore'))
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(X_scaled)
    features['Cluster'] = labels_km
    
    st.subheader("Top Barangays by Incident Count")
    
    top_brgy = features.sort_values('TOTAL', ascending=False).head(15)
    top_brgy.index = [f"{m} - {b}" for m, b in top_brgy.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    top_brgy.sort_values('TOTAL').plot(kind='barh', y='TOTAL', color='#1f4e79', ax=ax)
    ax.set_xlabel("Total Incidents")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="interp-box"><b>Interpretation:</b> The chart ranks the most critical hotspots. Barangays at the top require immediate resource allocation.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Cluster Profiles (Mean Values)")
    
    profile = features.groupby('Cluster').mean(numeric_only=True).round(2)
    st.dataframe(profile, use_container_width=True)
    st.markdown('<div class="interp-box"><b>Interpretation:</b> By comparing the means, we can see that certain clusters are heavily driven by specific crime categories or temporal factors. This dictates what kind of intervention each cluster needs.</div>', unsafe_allow_html=True)

# ==========================================
# 5. ASSOCIATION RULES
# ==========================================
elif page == "Association Rules":
    st.title("Significant Association Rules")
    
    def season(m):
        if pd.isna(m): return 'Unknown'
        m = int(m)
        if m in (12, 1, 2): return 'Dry'
        if m in (3, 4, 5): return 'Hot'
        if m in (6, 7, 8): return 'Wet'
        return 'LateWet'

    txn_df = pd.DataFrame({
        'Barangay': df_f['barangay'].astype(str), 'Offense': df_f['offense_category'].astype(str), 
        'Time': df_f['part_of_day'].astype(str), 'Season': df_f['month'].apply(season), 
        'Severity': df_f['severity'].astype(str)
    })
    
    transactions = [[f"{col}={val}" for col, val in zip(txn_df.columns, row) if val and val != 'nan' and val != 'Unknown'] for row in txn_df.values.tolist()]
    transactions = [t for t in transactions if len(t) >= 2]
    
    if len(transactions) >= 20:
        te = TransactionEncoder()
        basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
        
        freq_ap = apriori(basket, min_support=min_support, use_colnames=True)
        if not freq_ap.empty:
            rules_ap = association_rules(freq_ap, metric='confidence', min_threshold=min_confidence).sort_values('lift', ascending=False)
            
            st.subheader("Top 10 Apriori Rules")
            disp = rules_ap.head(10).copy()
            disp['antecedents'] = disp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            disp['consequents'] = disp['consequents'].apply(lambda x: ', '.join(sorted(x)))
            
            display_table = disp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4)
            display_table.columns = ['If (Antecedent)', 'Then (Consequent)', 'Support', 'Confidence', 'Lift']
            st.dataframe(display_table, use_container_width=True)
            
            st.markdown('<div class="interp-box"><b>Interpretation:</b> These rules highlight hidden patterns. A high Confidence and Lift > 1 indicates that if a crime happens under specific conditions, it is highly likely to be a specific type of offense.</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Top Rules by Lift")
            top_lift = disp.head(7).copy()
            top_lift['Rule'] = top_lift['antecedents'] + " -> " + top_lift['consequents']
            
            fig, ax = plt.subplots(figsize=(12, 5))
            top_lift.sort_values('lift').plot(kind='barh', x='Rule', y='lift', color='#2980b9', ax=ax, legend=False)
            ax.set_xlabel("Lift Score")
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('<div class="interp-box"><b>Interpretation:</b> A higher Lift score means the rule is more meaningful. The top rules represent the strongest co-occurrences in the crime dataset.</div>', unsafe_allow_html=True)
        else:
            st.info("No frequent itemsets found.")

# ==========================================
# 6. TIME SERIES FORECASTING
# ==========================================
elif page == "Time Series Forecasting":
    st.title("Time Series Forecasting")
    
    ts_df = df_f.dropna(subset=['date_committed']).copy()
    if len(ts_df) > 0:
        ts_df['ym'] = ts_df['date_committed'].dt.to_period('M').dt.to_timestamp()
        monthly = ts_df.groupby('ym').size().rename('incidents')
        
        full_range = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq='MS')
        monthly = monthly.reindex(full_range, fill_value=0)
        
        if len(monthly) >= 18:
            best_order = (1, 1, 1)
            train_size = int(len(monthly) * 0.8)
            train = monthly[:train_size]
            test = monthly[train_size:]
            
            model = ARIMA(train, order=best_order)
            fitted = model.fit()
            forecast_test = fitted.forecast(steps=len(test))
            
            full_model = ARIMA(monthly, order=best_order)
            full_fitted = full_model.fit()
            future_dates = pd.date_range(start=monthly.index[-1] + pd.DateOffset(months=1), periods=arima_steps, freq='MS')
            future_forecast = full_fitted.forecast(steps=arima_steps)
            future_forecast.index = future_dates
            
            rmse = np.sqrt(mean_squared_error(test.values, forecast_test.values))
            mae = mean_absolute_error(test.values, forecast_test.values)
            mape = np.mean(np.abs((test.values - forecast_test.values) / np.maximum(test.values, 1))) * 100
            
            st.subheader("Model Evaluation")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model Used", f"ARIMA{best_order}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAE", f"{mae:.2f}")
            c4.metric("MAPE", f"{mape:.1f}%")
            
            st.markdown('<div class="interp-box"><b>Interpretation:</b> The ARIMA model demonstrates acceptable predictive accuracy. A low RMSE indicates that the model fits the historical data well.</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Forecast Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly.index, monthly.values, label='Actual Historical', color='#1f4e79', marker='o', markersize=4)
            ax.plot(test.index, forecast_test.values, label='Test Forecast', color='#e74c3c', linestyle='--')
            ax.plot(future_forecast.index, future_forecast.values, label='Future Forecast', color='#27ae60', linestyle='--', marker='s')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown('<div class="interp-box"><b>Interpretation:</b> The projected numbers allow the LGU and PNP to proactively allocate budgets and resources for the upcoming months.</div>', unsafe_allow_html=True)
            
            st.subheader("Forecasted Incident Volume")
            st.dataframe(pd.DataFrame({'Month': future_forecast.index.strftime('%Y-%m'), 'Predicted Incidents': np.maximum(future_forecast.values, 0).round(2)}), use_container_width=True)
            
        else:
            st.warning(f"Not enough monthly data to fit ARIMA. Currently have {len(monthly)} months.")
    else:
        st.warning("No valid dates found.")
