from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class Rule:
    """
    Horn-clause style rule:
      if all antecedents are true, then consequent becomes true.
    Example:
      antecedents=["emergency"] -> consequent="needs_priority"
    """

    antecedents: Tuple[str, ...]
    consequent: str


class LogicEngine:
    """
    Minimal forward-chaining engine implementing Modus Ponens:
      (A -> B) and A  =>  B
    with conjunction in antecedents.
    """

    def __init__(self, rules: Iterable[Rule] | None = None):
        self._rules: List[Rule] = list(rules or [])

    @staticmethod
    def default_rules() -> List[Rule]:
        return [
            Rule(("emergency",), "needs_priority"),
            Rule(("low_fuel",), "needs_priority"),
            Rule(("vip",), "needs_priority"),
            Rule(("international", "departure"), "needs_priority"),
            Rule(("bad_weather", "arrival"), "needs_priority"),
            # --- PDF-style core rule examples (Modus Ponens) ---
            # Eğer yakıt < %10 VE uçuş süresi > 2 saat ise -> Acil iniş önceliği ver
            Rule(("fuel_lt_10pct", "duration_gt_120min"), "emergency_landing_priority"),
            # Acil iniş önceliği -> needs_priority
            Rule(("emergency_landing_priority",), "needs_priority"),
        ]

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def infer(self, facts: Dict[str, bool]) -> Dict[str, bool]:
        """
        Returns a new dict with derived facts added.
        """
        derived = dict(facts)
        changed = True
        while changed:
            changed = False
            for rule in self._rules:
                if derived.get(rule.consequent) is True:
                    continue
                if all(derived.get(a) is True for a in rule.antecedents):
                    derived[rule.consequent] = True
                    changed = True
        return derived


def facts_from_flight(
    flight: Mapping[str, Any],
    *,
    fuel_pct_key: str = "fuel_pct",
    duration_min_key: str = "flight_duration_min",
) -> Dict[str, bool]:
    """
    Converts numeric flight fields into atomic boolean facts so that
    Modus Ponens rules can fire.

    Expected (minimal) numeric fields:
    - fuel_pct: 0..100
    - flight_duration_min: minutes

    Other fields are optional and passed-through if already boolean-like.
    """
    fuel_pct = float(flight.get(fuel_pct_key, 100.0))
    duration_min = float(flight.get(duration_min_key, 0.0))

    facts: Dict[str, bool] = {
        "fuel_lt_10pct": fuel_pct < 10.0,
        "duration_gt_120min": duration_min > 120.0,
    }

    # Pass-through common boolean flags if provided by the caller
    for k in [
        "emergency",
        "low_fuel",
        "vip",
        "international",
        "bad_weather",
        "arrival",
        "departure",
    ]:
        v = flight.get(k, None)
        if isinstance(v, (bool, int, float)):
            facts[k] = bool(v)

    # Normalize type if caller provides "type": "arrival"/"departure"
    if "type" in flight:
        t = str(flight["type"]).strip().lower()
        facts["arrival"] = t == "arrival"
        facts["departure"] = t == "departure"

    return facts
