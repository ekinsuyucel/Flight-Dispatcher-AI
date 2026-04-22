from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class MathModelConfig:
    # Fuel consumption (kg/min) baseline and penalties
    base_burn_kg_per_min: float = 45.0
    holding_penalty_per_min: float = 1.25
    taxi_penalty_per_min: float = 0.75

    # Risk model coefficients (logistic regression style)
    w0: float = -2.2
    w_weather: float = 1.1
    w_congestion: float = 0.9
    w_emergency: float = 2.0
    w_low_fuel: float = 1.4

    # Wind/delay model coefficients (logistic)
    wd0: float = -1.8
    wd_crosswind: float = 0.11  # per m/s
    wd_headwind: float = 0.04  # per m/s (headwind can also cause spacing constraints)
    wd_gust: float = 0.09  # per m/s
    wd_visibility: float = -0.8  # higher visibility reduces delays

    # Route generation defaults
    base_final_leg_km: float = 18.0
    base_turn_km: float = 10.0


class MathModel:
    """
    Matrix & probability flavored model:
    - Simple linear fuel model (vectorized)
    - Simple logistic risk probability
    """

    def __init__(self, config: MathModelConfig | None = None):
        self.config = config or MathModelConfig()

    # ----------------------------
    # Matrix (route) utilities
    # ----------------------------
    @staticmethod
    def _deg2rad(deg: float) -> float:
        return float(deg) * np.pi / 180.0

    @staticmethod
    def rotation_matrix_2d(theta_rad: float) -> np.ndarray:
        c = float(np.cos(theta_rad))
        s = float(np.sin(theta_rad))
        return np.array([[c, -s], [s, c]], dtype=float)

    @staticmethod
    def transform_points(points_xy: np.ndarray, *, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Applies affine transform: p' = R p + t
        points_xy: (N,2)
        R: (2,2) rotation
        t: (2,) translation
        """
        pts = np.asarray(points_xy, dtype=float)
        return (pts @ R.T) + np.asarray(t, dtype=float).reshape(1, 2)

    def landing_route_xy(
        self,
        *,
        runway_heading_deg: float,
        runway_threshold_xy_km: Tuple[float, float] = (0.0, 0.0),
        wind_from_deg: float = 0.0,
        wind_speed_mps: float = 0.0,
    ) -> np.ndarray:
        """
        Computes a simple 2D approach route using matrix transforms.

        - Base route is defined in runway-aligned coordinates (x along runway, y cross-runway).
        - Then rotated to runway heading and translated to runway threshold.
        - Wind introduces a small drift on the cross-runway axis.

        Returns points in km: shape (N,2)
        """
        c = self.config

        # Base route in runway frame: final -> turn -> intercept
        base = np.array(
            [
                [-c.base_final_leg_km, 0.0],
                [-c.base_turn_km, 2.5],
                [-c.base_turn_km * 0.6, 1.2],
                [-c.base_turn_km * 0.3, 0.5],
                [0.0, 0.0],  # runway threshold
            ],
            dtype=float,
        )

        # Wind drift approximation: crosswind component pushes lateral offset
        wind_dir = self._deg2rad(float(wind_from_deg))
        runway = self._deg2rad(float(runway_heading_deg))
        rel = wind_dir - runway
        crosswind = float(wind_speed_mps) * float(np.sin(rel))
        drift_km = 0.08 * crosswind  # heuristic scaling
        base[:, 1] += drift_km

        R = self.rotation_matrix_2d(runway)
        t = np.array([float(runway_threshold_xy_km[0]), float(runway_threshold_xy_km[1])], dtype=float)
        return self.transform_points(base, R=R, t=t)

    # ----------------------------
    # Probability (delay) model
    # ----------------------------
    def wind_components_mps(
        self,
        *,
        wind_from_deg: float,
        wind_speed_mps: float,
        runway_heading_deg: float,
    ) -> Dict[str, float]:
        """
        Decomposes wind into crosswind and headwind components (m/s).
        Positive headwind means wind coming from ahead of aircraft on runway heading.
        """
        wind_dir = self._deg2rad(float(wind_from_deg))
        runway = self._deg2rad(float(runway_heading_deg))
        rel = wind_dir - runway
        crosswind = float(wind_speed_mps) * float(np.sin(rel))
        headwind = float(wind_speed_mps) * float(np.cos(rel))
        return {"crosswind_mps": float(crosswind), "headwind_mps": float(headwind)}

    def delay_probability_due_to_wind(
        self,
        *,
        wind_from_deg: float,
        wind_speed_mps: float,
        gust_mps: float,
        runway_heading_deg: float,
        visibility_norm: float,
    ) -> float:
        """
        Models delay probability in [0,1] as a function of wind and visibility.
        visibility_norm: 0..1 (1 = very good visibility)
        """
        c = self.config
        comps = self.wind_components_mps(
            wind_from_deg=wind_from_deg,
            wind_speed_mps=wind_speed_mps,
            runway_heading_deg=runway_heading_deg,
        )
        x = np.array(
            [
                1.0,
                abs(comps["crosswind_mps"]),
                max(0.0, comps["headwind_mps"]),
                float(gust_mps),
                float(visibility_norm),
            ],
            dtype=float,
        )
        w = np.array([c.wd0, c.wd_crosswind, c.wd_headwind, c.wd_gust, c.wd_visibility], dtype=float)
        z = float(np.dot(w, x))
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 0.0, 1.0))

    def estimate_fuel_burn_kg(
        self,
        *,
        flight_type: str,
        minutes_to_event: float,
        expected_hold_min: float,
        expected_taxi_min: float,
    ) -> float:
        c = self.config
        # Basic linear model; arrivals tend to burn more in holding
        type_multiplier = 1.10 if flight_type == "arrival" else 0.95
        burn = (
            c.base_burn_kg_per_min * type_multiplier * float(minutes_to_event)
            + c.holding_penalty_per_min * float(expected_hold_min) * 60.0
            + c.taxi_penalty_per_min * float(expected_taxi_min) * 30.0
        )
        return max(0.0, float(burn))

    def risk_probability(
        self,
        *,
        weather_severity: float,
        congestion_level: float,
        emergency: bool,
        low_fuel: bool,
    ) -> float:
        """
        Returns probability in [0,1] using sigmoid(w^T x).
        """
        c = self.config
        x = np.array(
            [
                1.0,
                float(weather_severity),
                float(congestion_level),
                1.0 if emergency else 0.0,
                1.0 if low_fuel else 0.0,
            ],
            dtype=float,
        )
        w = np.array([c.w0, c.w_weather, c.w_congestion, c.w_emergency, c.w_low_fuel], dtype=float)
        z = float(np.dot(w, x))
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 0.0, 1.0))

    def score_flight(
        self,
        *,
        minutes_to_event: float,
        weather_severity: float,
        congestion_level: float,
        emergency: bool,
        low_fuel: bool,
        fuel_burn_kg: float,
        needs_priority: bool,
    ) -> Dict[str, float]:
        """
        Produces a numeric score bundle used by the optimizer.
        Higher is 'more urgent / more important' (so should go earlier).
        """
        risk = self.risk_probability(
            weather_severity=weather_severity,
            congestion_level=congestion_level,
            emergency=emergency,
            low_fuel=low_fuel,
        )
        time_urgency = 1.0 / max(1.0, float(minutes_to_event))
        priority_boost = 0.5 if needs_priority else 0.0

        # Heuristic: combine risk + urgency + priority; penalize high burn a bit
        score = 2.0 * risk + 1.5 * time_urgency + priority_boost - 0.0005 * float(fuel_burn_kg)
        return {
            "risk": float(risk),
            "time_urgency": float(time_urgency),
            "priority_boost": float(priority_boost),
            "score": float(score),
        }
