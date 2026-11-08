import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster,HeatMap,Fullscreen
from streamlit_folium import st_folium

def render_incident_map(d,max_points=2500):
    mapped=d[d["valid_coordinates"]].dropna(subset=["latitude","longitude"]).copy()
    # Default geographic sanity check is Philippines; remove this filter for datasets elsewhere.
    local=mapped[mapped.latitude.between(4.0,22.5)&mapped.longitude.between(116.0,127.5)].copy()
    use_ph=st.checkbox("Apply Philippines coordinate bounds",value=True)
    if not use_ph: local=mapped
    if local.empty:
        st.warning("No records with usable coordinates fall inside the selected geographic bounds. The map cannot position records without coordinates.")
        return
    if len(local)>max_points:
        st.info(f"Showing a reproducible sample of {max_points:,} from {len(local):,} mappable records to keep the map responsive.")
        local=local.sample(max_points,random_state=42)
    center=[local.latitude.median(),local.longitude.median()]
    fmap=folium.Map(location=center,zoom_start=11,tiles="OpenStreetMap",control_scale=True)
    Fullscreen(position="topright").add_to(fmap)
    cluster=MarkerCluster(name="Incident clusters",disableClusteringAtZoom=17).add_to(fmap)
    for r in local.itertuples():
        date=r.date.strftime("%Y-%m-%d") if pd.notna(r.date) else "Not recorded"
        hour=f"{int(r.hour):02d}:00" if pd.notna(r.hour) else "Not recorded"
        popup=f"<b>Area:</b> {r.location_label}<br><b>Offense type:</b> {r.offense_type}<br><b>Offense:</b> {r.offense}<br><b>Date:</b> {date}<br><b>Time:</b> {hour}"
        folium.CircleMarker([r.latitude,r.longitude],radius=4,color="#b42335",fill=True,fill_color="#e5484d",fill_opacity=.7,weight=1,popup=folium.Popup(popup,max_width=320),tooltip=str(r.location_label)).add_to(cluster)
    HeatMap(local[["latitude","longitude"]].values.tolist(),name="Density heat layer",radius=18,blur=14,min_opacity=.25).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap,use_container_width=True,height=620,returned_objects=[])
    st.caption(f"Displayed {len(local):,} mapped records. Filtered records without coordinates or outside geographic bounds are not shown. Clusters/heat intensity are descriptive only.")
