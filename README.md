# Crime Intelligence & Decision Support Dashboard

A modular Streamlit conversion of the Crime Occurrence Analysis notebook.

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload the Excel workbooks in the left sidebar. All non-empty sheets are consolidated.

## Modules
- `app.py`: dashboard shell, filters, tabs, KPI cards, exports.
- `modules/data_loader.py`: workbook ingestion, column normalization, preprocessing and caching.
- `modules/analytics.py`: filtering, KPIs, quality checks and decision brief.
- `modules/visualizations.py`: Plotly visualizations.
- `modules/map_view.py`: Folium/OSM interactive incident map, clustering and heat layer.
- `modules/forecasting.py`: on-demand Random Forest monthly area forecast and chronological validation.

## Performance choices
- `st.cache_data` caches workbook parsing and forecast results.
- Forecast training runs only when the user clicks the forecast button.
- Map points are sampled reproducibly above the configured threshold.
- Charts are rendered only when their tab is selected by Streamlit's normal tab rerun behavior; for very large data, apply sidebar filters first.
- Random Forest is reduced to 160 trees for interactive responsiveness.

## Expected fields
Date, time, barangay, offense type, offense, stage of felony, latitude and longitude. Common aliases are normalized.

## Important interpretation and privacy limits
- Counts are reported records, not population-adjusted crime rates.
- The map plots only supplied valid coordinates. It does not geocode missing locations.
- Heatmap intensity and marker clusters are descriptive, not statistically validated hotspots.
- Forecasts estimate reported monthly counts, not individual behavior. Relative tiers are based on the forecast distribution in the dataset.
- Victim and suspect columns are excluded from the dashboard's record preview and exports by default. Secure the source workbooks and deployment appropriately.
- Review reporting completeness, population/exposure, local context, and data governance before operational decisions.

## Deployment
For Streamlit Community Cloud, push this folder to a private or appropriately governed Git repository, set `app.py` as the main file, and deploy. For sensitive crime data, use an institutionally approved hosting environment with access control; do not publish identifiable records in a public app.
