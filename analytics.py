import pandas as pd
def filter_data(df,date_range,areas,types,sources):
    d=df[df["date"].notna()].copy()
    if date_range and len(date_range)==2:
        d=d[d["date"].dt.date.between(date_range[0],date_range[1])]
    if areas: d=d[d["location_label"].isin(areas)]
    if types: d=d[d["offense_type"].isin(types)]
    if sources: d=d[d["source_file"].isin(sources)]
    return d

def kpi_summary(d,raw):
    n=len(d); mapped=int(d["valid_coordinates"].sum())
    return {"records":n,"areas":d["location_label"].nunique(),"offense_types":d["offense_type"].nunique(),
            "mapped":mapped,"mapped_pct":mapped/n*100 if n else 0,
            "duplicates":int(d["possible_duplicate"].sum())}

def data_quality(df):
    return pd.DataFrame({"field":df.columns,"missing_count":[df[c].isna().sum() for c in df],
      "missing_percent":[round(df[c].isna().mean()*100,2) for c in df],
      "data_type":[str(df[c].dtype) for c in df]}).sort_values("missing_percent",ascending=False)

def monthly_counts(d):
    return d.groupby(d["date"].dt.to_period("M").dt.to_timestamp()).size().rename("records").reset_index(name="records").rename(columns={"date":"month"})

def top_counts(d,col,n=12):
    return d[col].value_counts().head(n).rename_axis(col).reset_index(name="records")

def decision_brief(d):
    if d.empty:return ["No records are available for the current filters."]
    items=[]
    peak=d["location_label"].value_counts()
    if len(peak): items.append(f"{peak.index[0]} has the highest recorded count in the selected period ({int(peak.iloc[0]):,} records). Validate against population, reporting coverage, and local context before prioritizing resources.")
    off=d["offense_type"].value_counts()
    if len(off): items.append(f"{off.index[0]} is the most frequently recorded offense category ({int(off.iloc[0]):,} records). Review its case details and prevention context; frequency is not equivalent to severity.")
    days=d["day_of_week"].value_counts()
    if len(days): items.append(f"{days.index[0]} has the highest recorded weekday count. Check whether this pattern persists across months and is not driven by reporting or recording schedules.")
    if d["hour"].notna().any():
        h=int(d["hour"].dropna().mode().iloc[0]); items.append(f"The most common recorded hour is {h:02d}:00. Treat this as a descriptive signal and check missing time values before scheduling interventions.")
    mapped=int(d["valid_coordinates"].sum())
    items.append(f"{mapped:,} of {len(d):,} filtered records have valid coordinates ({mapped/len(d)*100:.1f}%). Improve geocoding/data capture if geographic coverage is insufficient.")
    items.append("Suggested action: convene a short review with analysts and local stakeholders; verify unusual increases, assess data completeness, and document the rationale for any operational response.")
    return items
