import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.analytics import monthly_counts,top_counts

LAYOUT=dict(template="plotly_white",margin=dict(l=20,r=20,t=55,b=20),font=dict(family="Arial",color="#23344d"))
def show(fig,height=380):
    fig.update_layout(**LAYOUT,height=height)
    import streamlit as st
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})

def render_overview(d):
    left,right=__import__("streamlit").columns([1.4,1])
    with left:
        m=monthly_counts(d)
        if not m.empty: show(px.line(m,x="month",y="records",markers=True,title="Monthly reported occurrences"))
    with right:
        a=top_counts(d,"location_label",10).sort_values("records")
        show(px.bar(a,x="records",y="location_label",orientation="h",title="Top areas by recorded count"))
    st.markdown("**Reading the overview:** The monthly line shows changes in recorded volume. The area ranking identifies where the supplied dataset contains the most records; it does not adjust for population or reporting access.")

def render_temporal(d):
    c1,c2=__import__("streamlit").columns(2)
    with c1:
        day=d["day_of_week"].value_counts().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).fillna(0).reset_index()
        day.columns=["day","records"]; show(px.bar(day,x="day",y="records",title="Occurrences by weekday"))
    with c2:
        hours=d["hour"].dropna().astype(int).value_counts().reindex(range(24),fill_value=0).rename_axis("hour").reset_index(name="records")
        show(px.bar(hours,x="hour",y="records",title="Occurrences by recorded hour"))
    valid=d.dropna(subset=["hour"])
    if not valid.empty:
        mat=valid.pivot_table(index="day_of_week",columns="hour",values="offense",aggfunc="count",fill_value=0)
        mat=mat.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reindex(columns=range(24),fill_value=0)
        show(px.imshow(mat,aspect="auto",title="Day-of-week × hour heatmap",labels={"x":"Hour","y":"Weekday","color":"Records"}),440)
    st.caption("Heatmap cells show counts for each weekday/hour combination. Missing or inconsistently recorded times can bias apparent peaks.")

def render_offense(d):
    a,b=__import__("streamlit").columns(2)
    with a:
        top=top_counts(d,"offense_type",15).sort_values("records")
        show(px.bar(top,x="records",y="offense_type",orientation="h",title="Most frequently recorded offense types"),460)
    with b:
        stg=top_counts(d,"stage",10)
        show(px.pie(stg,names="stage",values="records",hole=.48,title="Recorded stage composition"),430)
    pareto=top_counts(d,"offense_type",15)
    if not pareto.empty:
        pareto["cumulative_percent"]=pareto.records.cumsum()/pareto.records.sum()*100
        fig=go.Figure()
        fig.add_bar(x=pareto.offense_type,y=pareto.records,name="Records")
        fig.add_scatter(x=pareto.offense_type,y=pareto.cumulative_percent,name="Cumulative %",yaxis="y2",mode="lines+markers")
        fig.update_layout(title="Pareto analysis of offense categories",yaxis2=dict(overlaying="y",side="right",range=[0,105],title="Cumulative %"),xaxis_tickangle=-30)
        show(fig,430)
    valid=d.dropna(subset=["date"])
    top_types=valid.offense_type.value_counts().head(10).index
    mix=valid[valid.offense_type.isin(top_types)].assign(month_start=lambda x:x.date.dt.to_period("M").dt.to_timestamp()).groupby(["month_start","offense_type"]).size().reset_index(name="records")
    if not mix.empty: show(px.area(mix,x="month_start",y="records",color="offense_type",title="Monthly offense mix"),430)
    st.caption("The Pareto view ranks frequency, not seriousness. Stage composition depends on source classification and should be checked against the applicable recording definitions.")

def render_area(forecast):
    import streamlit as st
    if forecast is not None and not forecast.empty:
        top=forecast.head(15).sort_values("predicted_count")
        show(px.bar(top,x="predicted_count",y="area",color="relative_risk_tier",orientation="h",title="Next-month estimated reported counts"),450)
