import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import TimeSeriesSplit

FEATURES=["lag_1","lag_2","lag_3","rolling_mean_3","month","year","month_sin","month_cos"]

@st.cache_data(show_spinner=False,ttl=3600)
def build_forecast(df):
    source=df[df.date.notna()].copy()
    if source.empty:return pd.DataFrame(),{},pd.DataFrame()
    source["area"]=source.location_label.astype(str)
    source["month_start"]=source.date.dt.to_period("M").dt.to_timestamp()
    counts=source.groupby(["area","month_start"]).size().rename("count").reset_index()
    months=pd.date_range(counts.month_start.min(),counts.month_start.max(),freq="MS")
    if len(months)<4:return pd.DataFrame(),{},pd.DataFrame()
    areas=sorted(counts.area.unique())
    panel=counts.pivot(index="month_start",columns="area",values="count").reindex(months).fillna(0)
    panel.index.name="month_start"
    rows=[]
    for area in panel:
        s=panel[area]
        for i in range(3,len(s)):
            dt=s.index[i]
            rows.append({"area":area,"month_start":dt,"target":float(s.iloc[i]),
             "lag_1":float(s.iloc[i-1]),"lag_2":float(s.iloc[i-2]),"lag_3":float(s.iloc[i-3]),
             "rolling_mean_3":float(s.iloc[i-3:i].mean()),"month":dt.month,"year":dt.year,
             "month_sin":np.sin(2*np.pi*dt.month/12),"month_cos":np.cos(2*np.pi*dt.month/12)})
    model_df=pd.DataFrame(rows).sort_values("month_start")
    if len(model_df)<10:return pd.DataFrame(),{},pd.DataFrame()
    unique=sorted(model_df.month_start.unique())
    cut=max(1,int(len(unique)*.8)); cutoff=unique[min(cut,len(unique)-1)]
    train=model_df[model_df.month_start<cutoff]; test=model_df[model_df.month_start>=cutoff]
    metrics={}
    if len(train)>=5 and not test.empty:
        model=RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1,max_features=.9)
        model.fit(train[FEATURES],train.target)
        pred=np.maximum(0,model.predict(test[FEATURES]))
        baseline=np.maximum(0,test.lag_1.to_numpy())
        metrics={"test_rows":len(test),"model_MAE":mean_absolute_error(test.target,pred),
         "model_RMSE":mean_squared_error(test.target,pred)**.5,
         "baseline_MAE":mean_absolute_error(test.target,baseline)}
    cv=[]
    if len(unique)>=5:
        splitter=TimeSeriesSplit(n_splits=min(4,len(unique)-1))
        for fold,(tr_idx,va_idx) in enumerate(splitter.split(unique),1):
            trm=np.array(unique)[tr_idx]; vam=np.array(unique)[va_idx]
            tr=model_df[model_df.month_start.isin(trm)]; va=model_df[model_df.month_start.isin(vam)]
            if len(tr)<5 or va.empty:continue
            m=RandomForestRegressor(n_estimators=120,min_samples_leaf=2,random_state=42,n_jobs=-1,max_features=.9)
            m.fit(tr[FEATURES],tr.target); p=np.maximum(0,m.predict(va[FEATURES]))
            cv.append({"fold":fold,"validation_from":str(pd.Timestamp(vam.min()).date()),"validation_to":str(pd.Timestamp(vam.max()).date()),"rows":len(va),"MAE":mean_absolute_error(va.target,p)})
    final=RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=42,n_jobs=-1,max_features=.9)
    final.fit(model_df[FEATURES],model_df.target)
    next_month=(panel.index.max()+pd.offsets.MonthBegin(1)).normalize()
    future=[]
    for area in panel:
        s=panel[area]
        if len(s)<3:continue
        future.append({"area":area,"month_start":next_month,"lag_1":float(s.iloc[-1]),"lag_2":float(s.iloc[-2]),"lag_3":float(s.iloc[-3]),"rolling_mean_3":float(s.iloc[-3:].mean()),"month":next_month.month,"year":next_month.year,"month_sin":np.sin(2*np.pi*next_month.month/12),"month_cos":np.cos(2*np.pi*next_month.month/12)})
    forecast=pd.DataFrame(future)
    if forecast.empty:return forecast,metrics,pd.DataFrame(cv)
    forecast["predicted_count"]=np.maximum(0,final.predict(forecast[FEATURES]))
    q50,q80=forecast.predicted_count.quantile([.5,.8]).tolist()
    forecast["relative_risk_tier"]=np.select([forecast.predicted_count>=q80,forecast.predicted_count>=q50],["Higher relative forecast","Moderate relative forecast"],default="Lower relative forecast")
    forecast=forecast.sort_values("predicted_count",ascending=False)
    return forecast[["area","month_start","predicted_count","relative_risk_tier","lag_1","rolling_mean_3"]],metrics,pd.DataFrame(cv)
