import streamlit as st
import pandas as pd
from modules.data_loader import load_uploaded_workbooks
from modules.analytics import filter_data, kpi_summary, data_quality, monthly_counts, top_counts, decision_brief
from modules.visualizations import render_overview, render_temporal, render_offense, render_area
from modules.map_view import render_incident_map
from modules.forecasting import build_forecast, evaluate_forecast

st.set_page_config(page_title="Crime Intelligence | Decision Support", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#f5f7fb 0%,#edf2f8 100%);}
[data-testid="stSidebar"] {background:#101d35;}
[data-testid="stSidebar"] * {color:#eef4ff !important;}
.block-container {padding-top:1.5rem; max-width:1500px;}
.hero {background:linear-gradient(110deg,#10213f,#183c65);padding:26px 30px;border-radius:18px;color:white;margin-bottom:20px;box-shadow:0 8px 28px #18304b22;}
.hero h1 {font-family:Manrope,sans-serif;font-size:30px;margin:0;color:white;}
.hero p {color:#d7e5f6;margin:6px 0 0 0;}
.metric {background:white;border:1px solid #e5ebf3;border-radius:14px;padding:16px 18px;box-shadow:0 3px 14px #172b4d0a;}
.metric-label {font-size:12px;color:#64748b;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.metric-value {font-size:27px;color:#14243d;font-weight:800;margin-top:5px;}
.metric-note {font-size:11px;color:#718096;margin-top:3px;}
.section-note {background:#eef4fb;border-left:4px solid #3975b8;padding:12px 15px;border-radius:8px;color:#31465f;font-size:13px;margin:8px 0 16px;}
div[data-testid="stTabs"] button {font-weight:700;}
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🛡️ Crime Intelligence & Decision Support</h1><p>Evidence-led monitoring of reported occurrences • Temporal patterns • Geographic concentration • Area-level planning</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## CONTROL CENTER")
    st.caption("Upload one or more source Excel workbooks. All sheets are consolidated.")
    uploads=st.file_uploader("Crime occurrence workbooks",type=["xlsx","xls"],accept_multiple_files=True)
    st.divider()
    st.caption("Performance settings")
    map_limit=st.select_slider("Maximum map points",options=[500,1000,2500,5000,10000],value=2500)
    st.caption("Map markers are sampled when records exceed this limit.")

if not uploads:
    st.info("Upload the Excel workbooks used in the analysis to initialize the dashboard.")
    st.markdown("**Expected fields (aliases are supported):** date/date committed, time/time committed, barangay, offense type, offense, stage of felony, latitude, longitude.")
    st.markdown("**Privacy:** victim and suspect descriptive fields are excluded from dashboard views and exports.")
    st.stop()

try:
    raw=load_uploaded_workbooks(uploads)
except Exception as e:
    st.error(f"Unable to read the uploaded workbooks: {e}")
    st.stop()

if raw.empty:
    st.warning("The uploaded workbooks contain no readable records.")
    st.stop()

with st.sidebar:
    st.markdown("## FILTERS")
    dates=raw["date"].dropna()
    if dates.empty:
        st.error("No valid dates found. Review date formats in source files.")
        st.stop()
    min_date,max_date=dates.min().date(),dates.max().date()
    date_range=st.date_input("Date range",value=(min_date,max_date),min_value=min_date,max_value=max_date)
    areas=sorted(raw["location_label"].dropna().astype(str).unique())
    selected_areas=st.multiselect("Barangay / area",areas,default=[])
    types=sorted(raw["offense_type"].dropna().astype(str).unique())
    selected_types=st.multiselect("Offense type",types,default=[])
    sources=sorted(raw["source_file"].dropna().astype(str).unique())
    selected_sources=st.multiselect("Source workbook",sources,default=[])

filtered=filter_data(raw,date_range,selected_areas,selected_types,selected_sources)
if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

k=kpi_summary(filtered,raw)
c=st.columns(5)
labels=[("Reported records",f"{k['records']:,}","Filtered dataset"),("Areas represented",f"{k['areas']:,}","Distinct recorded barangays"),("Offense categories",f"{k['offense_types']:,}","Distinct classifications"),("Valid coordinates",f"{k['mapped']:,}",f"{k['mapped_pct']:.1f}% of filtered records"),("Possible duplicates",f"{k['duplicates']:,}","Flagged for review; not removed")]
for col,(a,b,d) in zip(c,labels):
    with col: metric_card(a,b,d)

tabs=st.tabs(["Executive Overview","Time & Patterns","Offense Intelligence","Geographic Intelligence","Forecast & Decisions","Data Quality & Export"])

with tabs[0]:
    st.subheader("Executive overview")
    st.markdown('<div class="section-note">This page summarizes the records that match the current filters. Counts describe recorded reports—not population-adjusted crime rates or the true incidence of crime.</div>',unsafe_allow_html=True)
    render_overview(filtered)

with tabs[1]:
    st.subheader("Temporal intelligence")
    render_temporal(filtered)

with tabs[2]:
    st.subheader("Offense profile")
    render_offense(filtered)

with tabs[3]:
    st.subheader("Geographic intelligence")
    st.markdown('<div class="section-note">The map plots only valid coordinates supplied in the workbooks. Heat intensity indicates record concentration, not statistically confirmed crime risk.</div>',unsafe_allow_html=True)
    render_incident_map(filtered,map_limit)

with tabs[4]:
    st.subheader("Forecasting and operational decision support")
    st.markdown('<div class="section-note">Forecasts estimate next-month reported counts from historical monthly area counts. Relative tiers are prioritization aids—not predictions of individual offending or proof that an area is dangerous.</div>',unsafe_allow_html=True)
    if st.button("Run / refresh area-level forecast",type="primary"):
        with st.spinner("Training and validating the forecasting model..."):
            forecast,metrics,cv=build_forecast(raw)
            st.session_state["crime_forecast"]=forecast
            st.session_state["crime_metrics"]=metrics
            st.session_state["crime_cv"]=cv
    forecast=st.session_state.get("crime_forecast")
    metrics=st.session_state.get("crime_metrics")
    cv=st.session_state.get("crime_cv")
    if forecast is not None:
        if forecast.empty:
            st.warning("Insufficient monthly history to generate a forecast. At least four months are needed for lag features; more history is recommended.")
        else:
            if metrics:
                m1,m2,m3,m4=st.columns(4)
                with m1: metric_card("Holdout MAE",f"{metrics['model_MAE']:.2f}","Average absolute error")
                with m2: metric_card("Baseline MAE",f"{metrics['baseline_MAE']:.2f}","Persistence baseline")
                with m3: metric_card("Holdout RMSE",f"{metrics['model_RMSE']:.2f}","Penalizes larger errors")
                with m4: metric_card("Test rows",f"{metrics['test_rows']:,}","Chronological holdout")
                st.caption("Compare model MAE with baseline MAE. A lower model MAE is better; if it is not lower, the model has not demonstrated improvement over simply carrying forward the previous month's count.")
            render_area(forecast)
            st.dataframe(forecast, use_container_width=True, hide_index=True)
            if cv is not None and not cv.empty:
                st.markdown("**Time-series cross-validation**")
                st.dataframe(cv,use_container_width=True,hide_index=True)
    else:
        st.info("Forecasting is intentionally run on demand to keep initial dashboard loading fast.")
    st.markdown("#### Decision brief")
    for item in decision_brief(filtered):
        st.markdown(f"- {item}")

with tabs[5]:
    st.subheader("Data quality, provenance and export")
    q=data_quality(raw)
    st.markdown("#### Completeness by field")
    st.dataframe(q,use_container_width=True)
    st.markdown("#### Source workbook summary")
    src=raw.groupby("source_file",dropna=False).agg(records=("source_file","size"),valid_dates=("date","count"),valid_coordinates=("valid_coordinates","sum"),possible_duplicates=("possible_duplicate","sum")).reset_index()
    st.dataframe(src,use_container_width=True,hide_index=True)
    st.markdown("#### Filtered record preview")
    safe=filtered.drop(columns=[x for x in ["victim","suspect"] if x in filtered.columns],errors="ignore")
    st.dataframe(safe.head(500),use_container_width=True,hide_index=True)
    import io
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as writer:
        safe.to_excel(writer,index=False,sheet_name="filtered_records")
        q.to_excel(writer,index=False,sheet_name="data_quality")
        src.to_excel(writer,index=False,sheet_name="source_summary")
        filtered["offense_type"].value_counts().rename_axis("offense_type").reset_index(name="records").to_excel(writer,index=False,sheet_name="offense_summary")
        filtered["location_label"].value_counts().rename_axis("barangay").reset_index(name="records").to_excel(writer,index=False,sheet_name="area_summary")
        if st.session_state.get("crime_forecast") is not None and not st.session_state["crime_forecast"].empty:
            st.session_state["crime_forecast"].to_excel(writer,index=False,sheet_name="forecast")
    st.download_button("Download analysis workbook",data=out.getvalue(),file_name="crime_intelligence_outputs.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary")
