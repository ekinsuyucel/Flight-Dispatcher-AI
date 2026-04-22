from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dispatcher_agent import RationalFlightDispatcher


def radar_view(schedule: pd.DataFrame) -> go.Figure:
    """
    Simple 'Airport Radar' in XY plane (km):
    - Concentric circles are range rings
    - Points are aircraft; color indicates priority
    """
    df = schedule.copy()
    df["priority"] = np.where(df["needs_priority"], "priority", "normal")

    # If positions are not provided, synthesize them from minutes_to_event (deterministic)
    if "radar_x_km" not in df.columns or "radar_y_km" not in df.columns:
        rng = np.random.default_rng(42)
        angle = rng.uniform(0, 2 * np.pi, size=len(df))
        radius = np.clip(df["minutes_to_event"].to_numpy(dtype=float) / 3.0, 4.0, 40.0)
        df["radar_x_km"] = radius * np.cos(angle)
        df["radar_y_km"] = radius * np.sin(angle)

    max_r = float(max(40.0, np.sqrt((df["radar_x_km"] ** 2 + df["radar_y_km"] ** 2).max())))
    rings = [10, 20, 30, 40]
    rings = [r for r in rings if r <= max_r + 1e-6]

    fig = go.Figure()

    # Range rings
    for r in rings:
        theta = np.linspace(0, 2 * np.pi, 240)
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode="lines",
                line=dict(color="rgba(120,180,160,0.35)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Crosshair
    fig.add_trace(
        go.Scatter(
            x=[-max_r, max_r],
            y=[0, 0],
            mode="lines",
            line=dict(color="rgba(120,180,160,0.35)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-max_r, max_r],
            mode="lines",
            line=dict(color="rgba(120,180,160,0.35)", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Aircraft points
    for label, color in [("priority", "#ff4b4b"), ("normal", "#00c0ff")]:
        sub = df[df["priority"] == label]
        fig.add_trace(
            go.Scatter(
                x=sub["radar_x_km"],
                y=sub["radar_y_km"],
                mode="markers",
                name="Öncelikli" if label == "priority" else "Normal",
                marker=dict(
                    size=np.clip(8 + 18 * sub["delay_prob_wind"].to_numpy(dtype=float), 8, 24),
                    color=color,
                    line=dict(color="rgba(10,20,20,0.6)", width=1),
                ),
                customdata=np.stack(
                    [
                        sub["flight_id"].to_numpy(),
                        sub["type"].to_numpy(),
                        sub["fuel_pct"].to_numpy(dtype=float),
                        sub["flight_duration_min"].to_numpy(dtype=float),
                        sub["risk_prob"].to_numpy(dtype=float),
                        sub["delay_prob_wind"].to_numpy(dtype=float),
                        sub["emergency_landing_priority"].to_numpy(dtype=bool),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Fuel: %{customdata[2]:.1f}%<br>"
                    "Süre: %{customdata[3]:.0f} dk<br>"
                    "Risk: %{customdata[4]:.3f}<br>"
                    "Rüzgar gecikme p: %{customdata[5]:.3f}<br>"
                    "Acil iniş önceliği: %{customdata[6]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="#071d1d",
        plot_bgcolor="#071d1d",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(
        title="X (km)",
        zeroline=False,
        gridcolor="rgba(120,180,160,0.18)",
        range=[-max_r, max_r],
        scaleanchor="y",
        scaleratio=1,
        showspikes=False,
        color="rgba(210,255,240,0.85)",
    )
    fig.update_yaxes(
        title="Y (km)",
        zeroline=False,
        gridcolor="rgba(120,180,160,0.18)",
        range=[-max_r, max_r],
        showspikes=False,
        color="rgba(210,255,240,0.85)",
    )
    return fig


def gantt_view(schedule: pd.DataFrame) -> go.Figure:
    df = schedule.copy()
    df["task"] = df["flight_id"] + " (" + df["type"] + ")"
    df["start"] = df["scheduled_min"]
    df["finish"] = df["scheduled_min"] + 2.5
    df["priority"] = np.where(df["needs_priority"], "priority", "normal")

    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="task",
        color="priority",
        hover_data={
            "minutes_to_event": True,
            "scheduled_min": True,
            "lateness_min": True,
            "risk_prob": ":.3f",
            "fuel_burn_kg": ":.1f",
            "utility_score": ":.3f",
        },
        color_discrete_map={"priority": "#d62728", "normal": "#1f77b4"},
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Simülasyon zamanı (dakika)")
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def scatter_view(schedule: pd.DataFrame) -> go.Figure:
    df = schedule.copy()
    df["priority"] = np.where(df["needs_priority"], "priority", "normal")
    fig = px.scatter(
        df,
        x="scheduled_min",
        y="risk_prob",
        color="priority",
        size="fuel_burn_kg",
        hover_data=["flight_id", "type", "minutes_to_event", "lateness_min", "utility_score"],
        color_discrete_map={"priority": "#d62728", "normal": "#1f77b4"},
    )
    fig.update_xaxes(title="Atanan slot (dakika)")
    fig.update_yaxes(title="Risk olasılığı")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10))
    return fig


def main() -> None:
    st.set_page_config(page_title="Rational Flight Dispatcher", layout="wide")
    st.title("Rational Flight Dispatcher")
    st.caption("Kurallar (Modus Ponens) + olasılık/yakıt modeli + Hill Climbing ile iniş/kalkış sıralama simülasyonu.")

    with st.sidebar:
        st.header("Senaryo")
        weather_mode = st.selectbox("Hava durumu", ["Açık", "Fırtınalı"], index=0)
        n = st.slider("Uçuş sayısı", min_value=6, max_value=40, value=18, step=1)
        separation = st.slider("Minimum ayrım (dk)", min_value=1.5, max_value=6.0, value=2.5, step=0.5)
        seed = st.number_input("Rastgelelik tohumu", min_value=0, max_value=10_000, value=11, step=1)

        st.divider()
        st.header("Hill Climbing")
        max_iters = st.slider("Max iter", min_value=200, max_value=6000, value=2000, step=200)
        neighbor_samples = st.slider("Komşu örnek sayısı", min_value=10, max_value=200, value=60, step=5)

    dispatcher = RationalFlightDispatcher(random_seed=int(seed))
    dispatcher.config.separation_min = float(separation)
    dispatcher.optimizer.config.max_iters = int(max_iters)
    dispatcher.optimizer.config.neighbor_samples = int(neighbor_samples)

    flights = dispatcher.generate_flights(int(n))
    # Weather scenario control:
    # - Fırtınalı => mantık kurallarında bad_weather tetiklenir ve rüzgar/visibility kötüleşir
    if weather_mode == "Fırtınalı":
        flights["bad_weather"] = True
        flights["weather_severity"] = np.clip(flights["weather_severity"] + 0.55, 0.0, 1.0)
        flights["congestion_level"] = np.clip(flights["congestion_level"] + 0.20, 0.0, 1.0)
        flights["wind_speed_mps"] = np.clip(flights["wind_speed_mps"] + 10.0, 0.0, 35.0)
        flights["gust_mps"] = np.clip(flights["gust_mps"] + 8.0, 0.0, 25.0)
        flights["visibility_norm"] = np.clip(flights["visibility_norm"] - 0.45, 0.0, 1.0)
    else:
        flights["bad_weather"] = False
        flights["weather_severity"] = np.clip(flights["weather_severity"] * 0.55, 0.0, 1.0)
        flights["wind_speed_mps"] = np.clip(flights["wind_speed_mps"] * 0.45, 0.0, 35.0)
        flights["gust_mps"] = np.clip(flights["gust_mps"] * 0.35, 0.0, 25.0)
        flights["visibility_norm"] = np.clip(flights["visibility_norm"] + 0.15, 0.0, 1.0)

    schedule, objective_val = dispatcher.dispatch(flights)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Uçuş", f"{len(schedule)}")
    kpi2.metric("Öncelikli", f"{int(schedule['needs_priority'].sum())}")
    kpi3.metric("Ortalama risk", f"{schedule['risk_prob'].mean():.3f}")
    kpi4.metric("Rüzgar gecikme p", f"{schedule['delay_prob_wind'].mean():.3f}")
    kpi5.metric("Amaç (utility)", f"{objective_val:.3f}")

    tab0, tab1, tab2, tab3 = st.tabs(["Havalimanı Radarı", "Zaman Çizelgesi", "Risk & Yakıt", "Tablo"])
    with tab0:
        st.subheader("Havalimanı Radarı")
        st.caption("Nokta boyutu: rüzgâra bağlı gecikme olasılığı. Renk: mantık tabanlı öncelik.")
        st.plotly_chart(radar_view(schedule), use_container_width=True)
    with tab1:
        st.plotly_chart(gantt_view(schedule), use_container_width=True)
    with tab2:
        st.plotly_chart(scatter_view(schedule), use_container_width=True)
    with tab3:
        cols = [
            "schedule_slot",
            "flight_id",
            "type",
            "minutes_to_event",
            "fuel_pct",
            "flight_duration_min",
            "bad_weather",
            "scheduled_min",
            "lateness_min",
            "needs_priority",
            "emergency_landing_priority",
            "risk_prob",
            "delay_prob_wind",
            "fuel_burn_kg",
            "utility_score",
        ]
        st.dataframe(schedule[cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

