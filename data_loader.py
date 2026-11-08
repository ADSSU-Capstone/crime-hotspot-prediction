import io, re
import numpy as np
import pandas as pd
import streamlit as st

ALIASES={
"date_committed":"date","datecommitted":"date","date":"date","date_of_commission":"date",
"time_committed":"time","timecommitted":"time","time":"time",
"stages_of_felony":"stage","stage_of_felony":"stage","stageoffelony":"stage","stage":"stage",
"offense_type":"offense_type","offensetype":"offense_type","type_of_offense":"offense_type",
"latitude":"latitude","lat":"latitude","longitude":"longitude","lng":"longitude","lon":"longitude",
"barangay":"barangay","victims":"victim","victim":"victim","suspect":"suspect","offense":"offense"
}
def normalize_column(col):
    key=re.sub(r"[^a-z0-9]+","_",str(col).strip().lower()).strip("_")
    return ALIASES.get(key,key)

def parse_time(value):
    if pd.isna(value): return pd.NaT
    if hasattr(value,"hour") and hasattr(value,"minute"):
        try: return pd.to_datetime(f"{value.hour:02d}:{value.minute:02d}:{getattr(value,'second',0):02d}",format="%H:%M:%S")
        except Exception: pass
    if isinstance(value,(int,float)) and 0 <= float(value) < 1:
        seconds=int(round(float(value)*86400))%86400
        return pd.to_datetime(f"{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}",format="%H:%M:%S")
    return pd.to_datetime(str(value).strip(),errors="coerce")

@st.cache_data(show_spinner=False)
def _read_workbooks(payloads):
    frames=[]
    for filename,content in payloads:
        book=pd.ExcelFile(io.BytesIO(content))
        for sheet in book.sheet_names:
            part=pd.read_excel(book,sheet_name=sheet)
            if part.empty: continue
            part.columns=[normalize_column(c) for c in part.columns]
            part["source_file"]=filename
            part["source_sheet"]=sheet
            frames.append(part)
    if not frames: return pd.DataFrame()
    df=pd.concat(frames,ignore_index=True,sort=False)
    for col in ["barangay","date","time","stage","offense","offense_type","latitude","longitude"]:
        if col not in df: df[col]=np.nan
    for col in ["barangay","stage","offense","offense_type"]:
        df[col]=df[col].astype("string").str.strip().replace({"":pd.NA,"nan":pd.NA,"None":pd.NA})
    df["date"]=pd.to_datetime(df["date"],errors="coerce",dayfirst=False)
    df["time_parsed"]=df["time"].apply(parse_time)
    df["hour"]=df["time_parsed"].dt.hour
    df["day_of_week"]=df["date"].dt.day_name()
    df["day_num"]=df["date"].dt.dayofweek
    df["month"]=df["date"].dt.month
    df["year"]=df["date"].dt.year
    df["year_month"]=df["date"].dt.to_period("M").astype("string")
    df["quarter"]=df["date"].dt.quarter
    df["latitude"]=pd.to_numeric(df["latitude"],errors="coerce")
    df["longitude"]=pd.to_numeric(df["longitude"],errors="coerce")
    keys=[c for c in ["barangay","date","time","stage","offense","offense_type","latitude","longitude"] if c in df]
    df["possible_duplicate"]=df.duplicated(subset=keys,keep=False) if keys else False
    df["valid_coordinates"]=df["latitude"].between(-90,90)&df["longitude"].between(-180,180)
    df["location_label"]=df["barangay"].fillna("Unknown / missing barangay").astype(str)
    df["offense_type"]=df["offense_type"].fillna("Unknown / missing offense type").astype(str)
    df["offense"]=df["offense"].fillna("Unknown / missing offense").astype(str)
    df["stage"]=df["stage"].fillna("Unknown / missing stage").astype(str)
    return df

def load_uploaded_workbooks(uploaded):
    payloads=tuple((f.name,f.getvalue()) for f in uploaded)
    return _read_workbooks(payloads)
