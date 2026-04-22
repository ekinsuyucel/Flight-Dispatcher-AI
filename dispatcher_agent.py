from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from logic_engine import LogicEngine, facts_from_flight
from math_model import MathModel
from optimizer import HillClimbingOptimizer


FlightType = Literal["arrival", "departure"]


@dataclass
class DispatcherConfig:
    separation_min: float = 2.5  # minimum time gap between consecutive operations
    late_penalty_per_min: float = 0.08  # penalize pushing flights too late
    priority_reward: float = 0.4  # reward scheduling priority flights earlier


class RationalFlightDispatcher:
    """
    Rational agent (utility-based):
    - Uses a logic engine (Modus Ponens) to infer 'needs_priority'
    - Uses a math model to estimate risk & fuel and produce scores
    - Uses hill climbing to optimize the final sequence
    """

    def __init__(
        self,
        *,
        logic: LogicEngine | None = None,
        model: MathModel | None = None,
        optimizer: HillClimbingOptimizer | None = None,
        config: DispatcherConfig | None = None,
        random_seed: int = 11,
    ):
        self.logic = logic or LogicEngine(LogicEngine.default_rules())
        self.model = model or MathModel()
        self.optimizer = optimizer or HillClimbingOptimizer()
        self.config = config or DispatcherConfig()
        self._rng = np.random.default_rng(random_seed)

    def generate_flights(self, n: int = 18) -> pd.DataFrame:
        """
        Creates a toy scenario dataset for demos.
        """
        n = int(n)
        types: List[FlightType] = [("arrival" if i % 2 == 0 else "departure") for i in range(n)]

        df = pd.DataFrame(
            {
                "flight_id": [f"F{i:03d}" for i in range(n)],
                "type": types,
                "minutes_to_event": self._rng.integers(10, 120, size=n).astype(float),
                # --- fields used by Modus Ponens numeric rules ---
                "fuel_pct": self._rng.uniform(5.0, 85.0, size=n),  # 0..100
                "flight_duration_min": self._rng.integers(35, 320, size=n).astype(float),
                "international": self._rng.random(n) < 0.35,
                "vip": self._rng.random(n) < 0.10,
                "emergency": self._rng.random(n) < 0.06,
                "low_fuel": self._rng.random(n) < 0.14,
                "bad_weather": self._rng.random(n) < 0.25,
                "weather_severity": self._rng.uniform(0.0, 1.0, size=n),
                "congestion_level": self._rng.uniform(0.0, 1.0, size=n),
                "expected_hold_min": self._rng.uniform(0.0, 18.0, size=n),
                "expected_taxi_min": self._rng.uniform(4.0, 22.0, size=n),
                # --- wind fields used by delay model / radar ---
                "runway_heading_deg": self._rng.choice([36.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], size=n),
                "wind_from_deg": self._rng.uniform(0.0, 360.0, size=n),
                "wind_speed_mps": self._rng.uniform(0.0, 18.0, size=n),
                "gust_mps": self._rng.uniform(0.0, 12.0, size=n),
                "visibility_norm": self._rng.uniform(0.4, 1.0, size=n),
            }
        )
        return df

    def _infer_priority(self, row: pd.Series) -> bool:
        facts = facts_from_flight(row)
        inferred = self.logic.infer(facts)
        return bool(inferred.get("needs_priority", False))

    def evaluate(self, flights: pd.DataFrame) -> pd.DataFrame:
        df = flights.copy()
        # Keep derived facts visible for the dashboard
        inferred_flags = df.apply(lambda r: self.logic.infer(facts_from_flight(r)), axis=1)
        df["needs_priority"] = inferred_flags.apply(lambda d: bool(d.get("needs_priority", False)))
        df["emergency_landing_priority"] = inferred_flags.apply(
            lambda d: bool(d.get("emergency_landing_priority", False))
        )

        fuel = []
        scores = []
        risk = []
        delay_prob = []
        for _idx, r in df.iterrows():
            burn = self.model.estimate_fuel_burn_kg(
                flight_type=r["type"],
                minutes_to_event=float(r["minutes_to_event"]),
                expected_hold_min=float(r["expected_hold_min"]),
                expected_taxi_min=float(r["expected_taxi_min"]),
            )
            bundle = self.model.score_flight(
                minutes_to_event=float(r["minutes_to_event"]),
                weather_severity=float(r["weather_severity"]),
                congestion_level=float(r["congestion_level"]),
                emergency=bool(r["emergency"]),
                low_fuel=bool(r["low_fuel"]),
                fuel_burn_kg=float(burn),
                needs_priority=bool(r["needs_priority"]),
            )
            dp = self.model.delay_probability_due_to_wind(
                wind_from_deg=float(r["wind_from_deg"]),
                wind_speed_mps=float(r["wind_speed_mps"]),
                gust_mps=float(r["gust_mps"]),
                runway_heading_deg=float(r["runway_heading_deg"]),
                visibility_norm=float(r["visibility_norm"]),
            )
            fuel.append(burn)
            scores.append(bundle["score"])
            risk.append(bundle["risk"])
            delay_prob.append(dp)

        df["fuel_burn_kg"] = np.array(fuel, dtype=float)
        df["risk_prob"] = np.array(risk, dtype=float)
        df["delay_prob_wind"] = np.array(delay_prob, dtype=float)
        df["utility_score"] = np.array(scores, dtype=float)
        return df

    def _objective_factory(self, df: pd.DataFrame):
        c = self.config

        base_minutes = df["minutes_to_event"].to_numpy(dtype=float)
        score = df["utility_score"].to_numpy(dtype=float)
        priority = df["needs_priority"].to_numpy(dtype=bool)

        def objective(order: List[int]) -> float:
            # Build scheduled start times with fixed separation
            t = 0.0
            util = 0.0
            for pos, idx in enumerate(order):
                scheduled = t
                desired = float(base_minutes[idx])

                lateness = max(0.0, scheduled - desired)
                early_bonus = max(0.0, desired - scheduled) * 0.01

                util += float(score[idx])
                util += c.priority_reward * (1.0 / (1.0 + pos)) if bool(priority[idx]) else 0.0
                util -= c.late_penalty_per_min * lateness
                util += early_bonus

                t += c.separation_min

            return float(util)

        return objective

    def dispatch(self, flights: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        df = self.evaluate(flights)

        # Greedy initial order: higher utility first, then earlier deadlines
        initial = (
            df.sort_values(["utility_score", "minutes_to_event"], ascending=[False, True])
            .reset_index(drop=True)
            .copy()
        )

        initial_order = list(range(len(initial)))
        objective = self._objective_factory(initial)
        best_order, best_val = self.optimizer.optimize(initial_order, objective)

        scheduled = initial.iloc[best_order].reset_index(drop=True).copy()
        scheduled["schedule_slot"] = np.arange(len(scheduled), dtype=int)
        scheduled["scheduled_min"] = scheduled["schedule_slot"].astype(float) * float(self.config.separation_min)
        scheduled["lateness_min"] = np.maximum(0.0, scheduled["scheduled_min"] - scheduled["minutes_to_event"])
        return scheduled, float(best_val)
