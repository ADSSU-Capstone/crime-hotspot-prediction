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
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="Crime Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #7B241C 0%, #C0392B 100%);
        padding: 1.3rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card h2 { color: white; margin: 0; font-size: 1.9rem; }
    .metric-card p { color: white; margin: 0; opacity: 0.9; }
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
    'stages of felony': 'stage_of_felony', 'stageoffelony': 'stage_of_felony',
    'offense': 'offense', 'offense type': 'offense_type', 'offensetype': 'offense_type',
    'victims status': 'victim_status', 'victim': 'victim', 'victims age': 'victim_age',
    'victims gender': 'victim_gender', 'victims occupation': 'victim_occupation',
    'suspects status': 'suspect_status', 'suspect': 'suspect', 'suspects age': 'suspect_age',
    'suspects gender': 'suspect_gender', 'suspects occupation': 'suspect_occupation',
    'latitude': 'latitude', 'longitude': 'longitude', 'lat': 'latitude', 'lng': 'longitude',
    'municipal': 'municipality', 'province': 'province', 'incident type': 'incident_type',
    'incidenttype': 'incident_type', 'firearms caliber': 'firearms_caliber',
    'firearms kind': 'firearms_kind', 'firearms make': 'firearms_make', 'firearms stat': 'firearms_status',
    'facaliber': 'firearms_caliber', 'fakind': 'firearms_kind', 'famake': 'firearms_make',
    'fastatus': 'firearms_status', 'drug involved': 'drug_involved', 'druginvolved': 'drug_involved',
    'vehicle kind': 'vehicle_kind', 'vehiclekind': 'vehicle_kind',
    'vehiclechassisno': 'vehicle_chassis_no', 'vehicle chassis no': 'vehicle_chassis_no', 'narrative': 'narrative'
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

def _extract_gender(val):
    if pd.isna(val): return None
    for token in str(val).replace('(', '').replace(')', '').split('/'):
        if token.strip().lower() in ('male', 'female', 'm', 'f'):
            return 'Male' if token.strip().lower().startswith('m') else 'Female'
    return None

def _extract_age(val):
    if pd.isna(val): return np.nan
    for token in str(val).replace('(', '').replace(')', '').split('/'):
        if token.strip().isdigit() and 0 < int(token) < 120:
            return int(token)
    return np.nan

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
    
    if 'offense' in df.columns:
        df['offense_category'] = df['offense'].apply(categorize_offense)
    else:
        df['offense_category'] = 'Other'

    df['victim_gender'] = df['victim'].apply(_extract_gender) if 'victim' in df.columns else None
    df['victim_age'] = df['victim'].apply(_extract_age) if 'victim' in df.columns else np.nan
    df['suspect_gender'] = df['suspect'].apply(_extract_gender) if 'suspect' in df.columns else None
    df['suspect_age'] = df['suspect'].apply(_extract_age) if 'suspect' in df.columns else np.nan

    def severity_label(row):
        v = str(row.get('victim', '')).lower()
        if 'killed' in v or 'deceased' in v or 'found dead' in v: return 'High'
        if 'hospitalized' in v or 'injured' in v or 'wounded' in v: return 'Medium'
        return 'Low'
    
    df['severity'] = df.apply(severity_label, axis=1)
    for col in ['latitude', 'longitude', 'victim_age', 'suspect_age']:
        df[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else np.nan
        
    return df[df['barangay'].notna() & (df['barangay'] != 'NAN') & (df['barangay'] != '')].copy()

df_raw = load_all_data()
if df_raw is None:
    st.error("Dataset files not found. Please ensure the 4 Excel files are in the root directory.")
    st.stop()
    
df = preprocess(df_raw)

st.sidebar.header("Data Filters")
selected_munis = st.sidebar.multiselect("Municipality", options=sorted(df['_source_municipality'].unique()), default=sorted(df['_source_municipality'].unique()))
year_min = int(df['year'].min()) if df['year'].notna().any() else 2020
year_max = int(df['year'].max()) if df['year'].notna().any() else 2025
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

k_clusters = st.sidebar.slider("K-Means Clusters", 2, 10, 4)
dbscan_eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 3.0, 0.5, 0.1)
dbscan_minpts = st.sidebar.slider("DBSCAN Min Samples", 2, 20, 5)
min_support = st.sidebar.slider("Association Rule Support", 0.005, 0.20, 0.02, 0.005)
min_confidence = st.sidebar.slider("Association Rule Confidence", 0.1, 1.0, 0.4, 0.05)
arima_steps = st.sidebar.slider("Forecast Horizon (Months)", 3, 24, 12)

df_f = df[(df['_source_municipality'].isin(selected_munis)) & (df['year'].between(year_range[0], year_range[1]))].copy()
if len(df_f) < 20:
    st.warning("Insufficient records for analysis. Adjust filters.")
    st.stop()

tabs = st.tabs(["Overview", "Temporal Patterns", "Clustering", "Association Rules", "Forecasting"])

with tabs[0]:
    st.subheader("Incident Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><p>Total Incidents</p><h2>{len(df_f):,}</h2></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><p>Municipalities</p><h2>{df_f["_source_municipality"].nunique()}</h2></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><p>Barangays</p><h2>{df_f["barangay"].nunique()}</h2></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><p>High Severity</p><h2>{(df_f["severity"]=="High").sum():,}</h2></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f['offense_category'].value_counts().plot(kind='bar', color='#7B241C', ax=ax, edgecolor='black')
        ax.set_title("Incidents by Category")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f['severity'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#e74c3c', '#e67e22', '#27ae60'], ax=ax)
        ax.set_title("Severity Distribution")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()

with tabs[1]:
    st.subheader("Temporal Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f.groupby('year').size().plot(kind='line', marker='o', color='#7B241C', ax=ax)
        ax.set_title("Incidents by Year")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        df_f.groupby('part_of_day').size().plot(kind='bar', color='#C0392B', ax=ax, edgecolor='black')
        ax.set_title("Incidents by Time of Day")
        ax.set_ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.subheader("Top Barangays")
    top_brgy = df_f.groupby(['_source_municipality', 'barangay']).size().sort_values(ascending=False).head(15)
    top_brgy.index = [f"{m} - {b}" for m, b in top_brgy.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    top_brgy.sort_values().plot(kind='barh', color='#7B241C', ax=ax, edgecolor='black')
    ax.set_xlabel("Incidents")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tabs[2]:
    st.subheader("Hotspot Clustering")
    
    offense_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='offense_category', values='offense', aggfunc='count', fill_value=0)
    time_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='part_of_day', values='offense', aggfunc='count', fill_value=0)
    time_pivot.columns = [f"TOD_{c}" for c in time_pivot.columns]
    sev_pivot = df_f.pivot_table(index=['_source_municipality', 'barangay'], columns='severity', values='offense', aggfunc='count', fill_value=0)
    sev_pivot.columns = [f"SEV_{c}" for c in sev_pivot.columns]
    total = df_f.groupby(['_source_municipality', 'barangay']).size().rename('TOTAL')
    geo = df_f.groupby(['_source_municipality', 'barangay']).agg(lat=('latitude', 'mean'), lng=('longitude', 'mean'))
    
    features = pd.concat([offense_pivot, time_pivot, sev_pivot, total, geo], axis=1).fillna(0)
    features = features[features['TOTAL'] > 0]
    
    if len(features) >= k_clusters * 2:
        X_scaled = StandardScaler().fit_transform(features.drop(columns=['lat', 'lng'], errors='ignore'))
        
        st.markdown("### K-Means Clustering")
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)
        
        sil_km = silhouette_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0
        dbi_km = davies_bouldin_score(X_scaled, labels_km) if len(set(labels_km)) > 1 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette Score", f"{sil_km:.4f}")
        c2.metric("Davies-Bouldin Index", f"{dbi_km:.4f}")
        c3.metric("Clustered Barangays", len(features))
        
        features_km = features.copy()
        features_km['CLUSTER_KM'] = labels_km
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_km, cmap='tab10', s=60, alpha=0.8, edgecolors='black')
        plt.colorbar(sc, label='Cluster')
        ax.set_title(f"PCA Plot (K={k_clusters})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("**Top Barangays per Cluster:**")
        for c in sorted(features_km['CLUSTER_KM'].unique()):
            sub = features_km[features_km['CLUSTER_KM'] == c].sort_values('TOTAL', ascending=False).head(5)
            st.markdown(f"**Cluster {c}**")
            display_df = sub[['TOTAL']].copy()
            display_df.index = [f"{m} - {b}" for m, b in display_df.index]
            st.dataframe(display_df, use_container_width=True)

        st.markdown("### DBSCAN Clustering")
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minpts)
        labels_db = dbscan.fit_predict(X_scaled)
        n_noise = (labels_db == -1).sum()
        
        c1, c2 = st.columns(2)
        c1.metric("Clusters Found", len(set(labels_db) - {-1}))
        c2.metric("Noise Points", n_noise)
        
        if n_noise > 0:
            st.markdown("**Noise Barangays:**")
            noise_df = features.copy()
            noise_df['CLUSTER_DB'] = labels_db
            noise_df = noise_df[noise_df['CLUSTER_DB'] == -1][['TOTAL']].sort_values('TOTAL', ascending=False)
            noise_df.index = [f"{m} - {b}" for m, b in noise_df.index]
            st.dataframe(noise_df, use_container_width=True)

with tabs[3]:
    st.subheader("Co-occurrence Patterns")
    
    def season(m):
        if pd.isna(m): return 'Unknown'
        m = int(m)
        if m in (12, 1, 2): return 'Season_Dry'
        if m in (3, 4, 5): return 'Season_Hot'
        if m in (6, 7, 8): return 'Season_Wet'
        return 'Season_LateWet'

    txn_df = pd.DataFrame({
        'MUNI': df_f['_source_municipality'].astype(str), 'BRGY': df_f['barangay'].astype(str),
        'OFFENSE': df_f['offense_category'].astype(str), 'TOD': df_f['part_of_day'].astype(str),
        'SEASON': df_f['month'].apply(season), 'SEVERITY': df_f['severity'].astype(str)
    })
    
    transactions = [[str(x) for x in row if x and x != 'nan' and x != 'Unknown'] for row in txn_df.values.tolist()]
    transactions = [t for t in transactions if len(t) >= 2]
    
    if len(transactions) >= 20:
        te = TransactionEncoder()
        basket = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Apriori**")
            freq_ap = apriori(basket, min_support=min_support, use_colnames=True)
            if not freq_ap.empty:
                rules_ap = association_rules(freq_ap, metric='confidence', min_threshold=min_confidence).sort_values('lift', ascending=False)
                if not rules_ap.empty:
                    disp = rules_ap.head(10).copy()
                    disp['antecedents'] = disp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                    disp['consequents'] = disp['consequents'].apply(lambda x: ', '.join(sorted(x)))
                    st.dataframe(disp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4), use_container_width=True)
                else:
                    st.write("No rules found.")
            else:
                st.write("No frequent itemsets.")

        with col2:
            st.markdown("**FP-Growth**")
            freq_fp = fpgrowth(basket, min_support=min_support, use_colnames=True)
            if not freq_fp.empty:
                rules_fp = association_rules(freq_fp, metric='confidence', min_threshold=min_confidence).sort_values('lift', ascending=False)
                if not rules_fp.empty:
                    disp = rules_fp.head(10).copy()
                    disp['antecedents'] = disp['antecedents'].apply(lambda x: ', '.join(sorted(x)))
                    disp['consequents'] = disp['consequents'].apply(lambda x: ', '.join(sorted(x)))
                    st.dataframe(disp[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(4), use_container_width=True)
                else:
                    st.write("No rules found.")
            else:
                st.write("No frequent itemsets.")

with tabs[4]:
    st.subheader("ARIMA Forecasting")
    
    ts_df = df_f.dropna(subset=['date_committed']).copy()
    if len(ts_df) > 0:
        ts_df['ym'] = ts_df['date_committed'].dt.to_period('M').dt.to_timestamp()
        monthly = ts_df.groupby('ym').size().rename('incidents')
        
        # Fix: ensure continuous timeline by reindexing
        full_range = pd.date_range(start=monthly.index.min(), end=monthly.index.max(), freq='MS')
        monthly = monthly.reindex(full_range, fill_value=0)
        
        # Need at least 18 months of data to train an ARIMA model properly
        if len(monthly) >= 18:
            # Fix: Force a stable ARIMA(1,1,1) to prevent flat-data crashes
            best_order = (1, 1, 1)
            
            train_size = int(len(monthly) * 0.8)
            train = monthly[:train_size]
            test = monthly[train_size:]
            
            # Fit model on train set
            model = ARIMA(train, order=best_order)
            fitted = model.fit()
            forecast_test = fitted.forecast(steps=len(test))
            
            # Fit model on full dataset for future forecast
            full_model = ARIMA(monthly, order=best_order)
            full_fitted = full_model.fit()
            future_dates = pd.date_range(start=monthly.index[-1] + pd.DateOffset(months=1), periods=arima_steps, freq='MS')
            future_forecast = full_fitted.forecast(steps=arima_steps)
            future_forecast.index = future_dates
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(test.values, forecast_test.values))
            mae = mean_absolute_error(test.values, forecast_test.values)
            mape = np.mean(np.abs((test.values - forecast_test.values) / np.maximum(test.values, 1))) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", f"ARIMA{best_order}")
            c2.metric("RMSE", f"{rmse:.2f}")
            c3.metric("MAE", f"{mae:.2f}")
            c4.metric("MAPE", f"{mape:.1f}%")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly.index, monthly.values, label='Actual', color='#7B241C', marker='o', markersize=3)
            ax.plot(test.index, forecast_test.values, label='Test Forecast', color='#2980b9', linestyle='--')
            ax.plot(future_forecast.index, future_forecast.values, label='Future Forecast', color='#27ae60', linestyle='--', marker='s')
            ax.set_title(f"ARIMA{best_order} Model")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown("**Forecasted Values**")
            st.dataframe(pd.DataFrame({'Month': future_forecast.index.strftime('%Y-%m'), 'Predicted Incidents': np.maximum(future_forecast.values, 0).round(2)}), use_container_width=True)
            
        else:
            st.warning(f"Not enough monthly data to fit ARIMA. Need at least 18 months of complete date records. Currently have {len(monthly)} months.")
    else:
        st.warning("No valid dates found in the dataset.")
