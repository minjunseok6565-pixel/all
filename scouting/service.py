from __future__ import annotations

"""Scouting service (DB-backed).

This module implements:
  - per-team scouting staff seeding (6~7 scouts per team)
  - monthly (end-of-month) scouting checkpoints

Key gameplay rules (as requested):
  - Scouting is 100% user-driven.
    If there are no ACTIVE scouting_assignments, this module must be a no-op.
  - Reports are written at *month end*.
  - If a scout was assigned within 14 days of the month end, they do NOT write
    a report for that month (insufficient info).
  - Scouting info improves over time via Bayesian/Kalman-style updates
    (uncertainty shrinks) but *bias remains*.
  - Scouts have specialties; axes improve at different speeds.
"""

import datetime as _dt
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import game_time
from league_repo import LeagueRepo
from ratings_2k import potential_grade_to_scalar

try:
    # Used to compute gameplay-relevant skill axes from attrs_json.
    from derived_formulas import compute_derived
except Exception:  # pragma: no cover
    compute_derived = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# JSON helpers (keep identical style with the rest of the project)
# -----------------------------------------------------------------------------

def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    s = str(value)
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Deterministic RNG seeding
# -----------------------------------------------------------------------------

def _stable_seed(*parts: object) -> int:
    """Stable seed across runs, independent of Python's hash randomization."""
    s = "|".join(str(p) for p in parts)
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


# -----------------------------------------------------------------------------
# Date helpers
# -----------------------------------------------------------------------------

def _parse_date_iso(value: Any, *, field: str) -> _dt.date:
    s = str(value)[:10]
    try:
        return _dt.date.fromisoformat(s)
    except Exception as e:
        raise ValueError(f"{field} must be ISO YYYY-MM-DD: got {value!r}") from e


def _month_floor(d: _dt.date) -> _dt.date:
    return _dt.date(int(d.year), int(d.month), 1)


def _add_one_month(d: _dt.date) -> _dt.date:
    y = int(d.year)
    m = int(d.month) + 1
    if m == 13:
        return _dt.date(y + 1, 1, 1)
    return _dt.date(y, m, 1)


def _month_end(month_start: _dt.date) -> _dt.date:
    nxt = _add_one_month(month_start)
    return nxt - _dt.timedelta(days=1)


def _infer_college_season_year_from_date(d: _dt.date) -> int:
    # Match college/service.py assumption: season starts in October.
    return int(d.year) if int(d.month) >= 10 else int(d.year - 1)


# -----------------------------------------------------------------------------
# Scouting axes
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AxisDef:
    key: str
    label: str
    # Baseline measurement noise for one day of observation (lower => easier to evaluate)
    base_meas_std: float
    # Initial uncertainty (stddev) when the scout starts evaluating the player
    init_sigma: float
    # Lower bound on uncertainty (bias remains even when sigma is low)
    sigma_floor: float
    # Lower bound on measurement noise (irreducible noise)
    meas_floor: float


# Canonical axes used in reports. Values are in 0..100 (higher is better).
AXES: Dict[str, AxisDef] = {
    "overall": AxisDef(
        key="overall",
        label="Overall",
        base_meas_std=18.0,
        init_sigma=22.0,
        sigma_floor=6.0,
        meas_floor=4.0,
    ),
    "athleticism": AxisDef(
        key="athleticism",
        label="Athleticism",
        base_meas_std=10.0,
        init_sigma=16.0,
        sigma_floor=4.0,
        meas_floor=2.5,
    ),
    "shooting": AxisDef(
        key="shooting",
        label="Shooting",
        base_meas_std=15.0,
        init_sigma=18.0,
        sigma_floor=5.0,
        meas_floor=3.0,
    ),
    "finishing": AxisDef(
        key="finishing",
        label="Finishing",
        base_meas_std=14.0,
        init_sigma=18.0,
        sigma_floor=5.0,
        meas_floor=3.0,
    ),
    "playmaking": AxisDef(
        key="playmaking",
        label="Playmaking",
        base_meas_std=16.0,
        init_sigma=19.0,
        sigma_floor=5.5,
        meas_floor=3.2,
    ),
    "defense": AxisDef(
        key="defense",
        label="Defense",
        base_meas_std=17.0,
        init_sigma=20.0,
        sigma_floor=6.0,
        meas_floor=3.5,
    ),
    "rebounding": AxisDef(
        key="rebounding",
        label="Rebounding",
        base_meas_std=13.0,
        init_sigma=18.0,
        sigma_floor=5.0,
        meas_floor=3.0,
    ),
    "upside": AxisDef(
        key="upside",
        label="Upside",
        base_meas_std=20.0,
        init_sigma=24.0,
        sigma_floor=7.0,
        meas_floor=4.5,
    ),
    "durability": AxisDef(
        key="durability",
        label="Durability",
        base_meas_std=24.0,
        init_sigma=28.0,
        sigma_floor=10.0,
        meas_floor=6.0,
    ),
    "character": AxisDef(
        key="character",
        label="Character",
        base_meas_std=26.0,
        init_sigma=30.0,
        sigma_floor=12.0,
        meas_floor=6.5,
    ),
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _clamp100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def _mean(values: Sequence[float], default: float = 50.0) -> float:
    vs = [float(v) for v in values if v is not None]
    if not vs:
        return float(default)
    return float(sum(vs) / float(len(vs)))


def _compute_true_axis_values(*, ovr: int, attrs: Mapping[str, Any]) -> Dict[str, float]:
    """Compute "true" axis values (0..100) from SSOT attrs_json.

    Notes:
      - We deliberately use gameplay-oriented derived metrics when available.
      - Some axes (upside/durability/character) come from extended SSOT keys.
    """
    out: Dict[str, float] = {}

    out["overall"] = _clamp100(float(ovr))

    # Derived metrics (if available)
    derived: Dict[str, float] = {}
    if compute_derived is not None:
        try:
            derived = compute_derived(dict(attrs)) or {}
        except Exception:
            derived = {}

    # Athleticism: first step + physical + endurance
    if derived:
        out["athleticism"] = _clamp100(
            0.50 * _safe_float(derived.get("FIRST_STEP"), 50.0)
            + 0.25 * _safe_float(derived.get("PHYSICAL"), 50.0)
            + 0.25 * _safe_float(derived.get("ENDURANCE"), 50.0)
        )
        out["shooting"] = _clamp100(
            0.40 * _safe_float(derived.get("SHOT_3_CS"), 50.0)
            + 0.30 * _safe_float(derived.get("SHOT_MID_CS"), 50.0)
            + 0.20 * _safe_float(derived.get("SHOT_TOUCH"), 50.0)
            + 0.10 * _safe_float(derived.get("SHOT_FT"), 50.0)
        )
        out["finishing"] = _clamp100(
            0.45 * _safe_float(derived.get("FIN_RIM"), 50.0)
            + 0.30 * _safe_float(derived.get("FIN_DUNK"), 50.0)
            + 0.25 * _safe_float(derived.get("FIN_CONTACT"), 50.0)
        )
        out["playmaking"] = _clamp100(
            _mean(
                [
                    _safe_float(derived.get("PASS_CREATE"), 50.0),
                    _safe_float(derived.get("PASS_SAFE"), 50.0),
                    _safe_float(derived.get("PNR_READ"), 50.0),
                    _safe_float(derived.get("HANDLE_SAFE"), 50.0),
                    _safe_float(derived.get("DRIVE_CREATE"), 50.0),
                ],
                default=50.0,
            )
        )
        out["defense"] = _clamp100(
            _mean(
                [
                    _safe_float(derived.get("DEF_POA"), 50.0),
                    _safe_float(derived.get("DEF_HELP"), 50.0),
                    _safe_float(derived.get("DEF_RIM"), 50.0),
                    _safe_float(derived.get("DEF_STEAL"), 50.0),
                    _safe_float(derived.get("DEF_POST"), 50.0),
                ],
                default=50.0,
            )
        )
        out["rebounding"] = _clamp100(
            _mean(
                [
                    _safe_float(derived.get("REB_OR"), 50.0),
                    _safe_float(derived.get("REB_DR"), 50.0),
                ],
                default=50.0,
            )
        )
    else:
        # Minimal fallback from base keys (still 0..100)
        out["athleticism"] = _mean(
            [
                _safe_float(attrs.get("Speed"), 50.0),
                _safe_float(attrs.get("Agility"), 50.0),
                _safe_float(attrs.get("Vertical"), 50.0),
                _safe_float(attrs.get("Strength"), 50.0),
                _safe_float(attrs.get("Stamina"), 50.0),
            ],
            default=50.0,
        )
        out["shooting"] = _mean(
            [
                _safe_float(attrs.get("Three-Point Shot"), 50.0),
                _safe_float(attrs.get("Mid-Range Shot"), 50.0),
                _safe_float(attrs.get("Free Throw"), 50.0),
                _safe_float(attrs.get("Shot IQ"), 50.0),
            ],
            default=50.0,
        )
        out["finishing"] = _mean(
            [
                _safe_float(attrs.get("Layup"), 50.0),
                _safe_float(attrs.get("Driving Dunk"), 50.0),
                _safe_float(attrs.get("Standing Dunk"), 50.0),
                _safe_float(attrs.get("Close Shot"), 50.0),
            ],
            default=50.0,
        )
        out["playmaking"] = _mean(
            [
                _safe_float(attrs.get("Ball Handle"), 50.0),
                _safe_float(attrs.get("Speed with Ball"), 50.0),
                _safe_float(attrs.get("Pass Vision"), 50.0),
                _safe_float(attrs.get("Pass Accuracy"), 50.0),
                _safe_float(attrs.get("Pass IQ"), 50.0),
            ],
            default=50.0,
        )
        out["defense"] = _mean(
            [
                _safe_float(attrs.get("Perimeter Defense"), 50.0),
                _safe_float(attrs.get("Interior Defense"), 50.0),
                _safe_float(attrs.get("Steal"), 50.0),
                _safe_float(attrs.get("Block"), 50.0),
                _safe_float(attrs.get("Help Defense IQ"), 50.0),
            ],
            default=50.0,
        )
        out["rebounding"] = _mean(
            [
                _safe_float(attrs.get("Offensive Rebound"), 50.0),
                _safe_float(attrs.get("Defensive Rebound"), 50.0),
                _safe_float(attrs.get("Hustle"), 50.0),
            ],
            default=50.0,
        )

    # Upside: Potential grade (A+..F) -> scalar -> 40..100-ish
    pot = attrs.get("Potential")
    pot_scalar = potential_grade_to_scalar(pot)
    out["upside"] = _clamp100(100.0 * _clamp01(float(pot_scalar)))

    # Durability: blend of "Overall Durability" + inverse injury frequency
    base_dur = _safe_float(attrs.get("Overall Durability"), 50.0)
    inj = attrs.get("I_InjuryFreq")
    try:
        inj_i = int(inj) if inj is not None else 5
    except Exception:
        inj_i = 5
    inj_i = int(max(1, min(10, inj_i)))
    inv_inj = 100.0 - float(inj_i - 1) * 10.0  # 1->100, 10->10
    out["durability"] = _clamp100(0.70 * base_dur + 0.30 * inv_inj)

    # Character: blend of mental keys (Ego is inverted)
    m_keys_pos = ["M_WorkEthic", "M_Coachability", "M_Ambition", "M_Adaptability", "M_Loyalty"]
    m_pos = [_safe_float(attrs.get(k), 60.0) for k in m_keys_pos]
    ego = _safe_float(attrs.get("M_Ego"), 55.0)
    ego_inv = 100.0 - ego
    out["character"] = _clamp100(_mean(m_pos + [ego_inv], default=60.0))

    # Ensure all known axes exist (fallback to 50)
    for k in AXES.keys():
        if k not in out:
            out[k] = 50.0
    return out


# -----------------------------------------------------------------------------
# Scout staff profiles (seeded, per team)
# -----------------------------------------------------------------------------


def _grade_from_0_100(v: float) -> str:
    x = float(v)
    if x >= 90:
        return "A+"
    if x >= 85:
        return "A"
    if x >= 80:
        return "A-"
    if x >= 75:
        return "B+"
    if x >= 70:
        return "B"
    if x >= 65:
        return "B-"
    if x >= 60:
        return "C+"
    if x >= 55:
        return "C"
    if x >= 50:
        return "C-"
    if x >= 45:
        return "D+"
    if x >= 40:
        return "D"
    return "F"


def _confidence_from_sigma(sigma: float) -> str:
    s = float(sigma)
    if s <= 5.0:
        return "high"
    if s <= 9.0:
        return "medium"
    return "low"


def _default_scout_templates(*, scouts_per_team: int) -> List[Dict[str, Any]]:
    """Return scout templates used by ensure_scouts_seeded()."""
    n = int(scouts_per_team)
    if n <= 0:
        return []

    # 7-role default staff. If n < 7 we truncate; if n > 7 we append generalists.
    base: List[Dict[str, Any]] = [
        {
            "specialty_key": "ATHLETICS",
            "display_name": "Tools Scout",
            "focus_axes": ["athleticism", "finishing"],
            "style_tags": ["practical", "tools-focused", "concise"],
            "acc": {"athleticism": 0.70, "finishing": 0.85},
            "learn": {"athleticism": 1.50, "finishing": 1.10},
            "bias": {"shooting": -2.0},
        },
        {
            "specialty_key": "SHOOTING",
            "display_name": "Shooting Specialist",
            "focus_axes": ["shooting"],
            "style_tags": ["detail-oriented", "shot-mechanics", "range"],
            "acc": {"shooting": 0.70},
            "learn": {"shooting": 1.60},
            "bias": {"defense": -2.0},
        },
        {
            "specialty_key": "DEFENSE",
            "display_name": "Defense Scout",
            "focus_axes": ["defense", "rebounding"],
            "style_tags": ["defense-first", "scheme", "matchups"],
            "acc": {"defense": 0.75, "rebounding": 0.85},
            "learn": {"defense": 1.35, "rebounding": 1.10},
            "bias": {"shooting": -1.5},
        },
        {
            "specialty_key": "PLAYMAKING",
            "display_name": "Playmaking Scout",
            "focus_axes": ["playmaking"],
            "style_tags": ["processing", "reads", "decision-making"],
            "acc": {"playmaking": 0.75},
            "learn": {"playmaking": 1.45},
            "bias": {"finishing": -1.0},
        },
        {
            "specialty_key": "MEDICAL",
            "display_name": "Medical Scout",
            "focus_axes": ["durability"],
            "style_tags": ["risk", "medical", "availability"],
            "acc": {"durability": 0.85},
            "learn": {"durability": 0.75},
            "bias": {"durability": -2.0},  # conservative by default
        },
        {
            "specialty_key": "CHARACTER",
            "display_name": "Character Scout",
            "focus_axes": ["character"],
            "style_tags": ["locker-room", "coachability", "intangibles"],
            "acc": {"character": 0.90},
            "learn": {"character": 0.65},
            "bias": {"character": -1.0},
        },
        {
            "specialty_key": "ANALYTICS",
            "display_name": "Analytics Scout",
            "focus_axes": ["overall", "upside"],
            "style_tags": ["projection", "probabilistic", "context"],
            "acc": {"overall": 0.85, "upside": 0.85},
            "learn": {"overall": 1.00, "upside": 0.85},
            "bias": {"athleticism": -1.0},
        },
    ]

    out: List[Dict[str, Any]] = []
    for i, t in enumerate(base):
        if i >= n:
            break
        out.append(t)

    # If n > 7, add generalists.
    extra = n - len(out)
    for j in range(extra):
        out.append(
            {
                "specialty_key": f"GENERAL_{j+1}",
                "display_name": "Regional Scout",
                "focus_axes": ["overall", "shooting", "defense"],
                "style_tags": ["generalist"],
                "acc": {"overall": 1.00, "shooting": 1.00, "defense": 1.00},
                "learn": {"overall": 0.90, "shooting": 0.90, "defense": 0.90},
                "bias": {},
            }
        )

    return out


def ensure_scouts_seeded(
    *,
    db_path: str,
    team_ids: Sequence[str],
    scouts_per_team: int = 7,
) -> Dict[str, Any]:
    """Ensure each team has a seeded scout staff (idempotent).

    This creates rows in scouting_scouts only.
    It does NOT create assignments; scouting remains fully user-driven.
    """
    if not db_path:
        raise ValueError("db_path is required")
    teams = [str(t).strip().upper() for t in (team_ids or []) if str(t).strip()]
    if not teams:
        return {"ok": True, "created": 0, "existing": 0, "teams": 0}

    templates = _default_scout_templates(scouts_per_team=int(scouts_per_team))
    if not templates:
        return {"ok": True, "created": 0, "existing": 0, "teams": len(teams)}

    now = game_time.now_utc_like_iso()

    created = 0
    existing = 0

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        with repo.transaction() as cur:
            for team_id in teams:
                for t in templates:
                    specialty_key = str(t.get("specialty_key") or "GENERAL").strip().upper()
                    display_name = str(t.get("display_name") or "Scout").strip()

                    scout_id = f"SCT_{team_id}_{specialty_key}"
                    row = cur.execute(
                        "SELECT scout_id FROM scouting_scouts WHERE scout_id=? LIMIT 1;",
                        (scout_id,),
                    ).fetchone()
                    if row:
                        existing += 1
                        continue

                    profile = {
                        "schema_version": 1,
                        "specialty_key": specialty_key,
                        "focus_axes": list(t.get("focus_axes") or []),
                        "acc_mult_by_axis": dict(t.get("acc") or {}),
                        "learn_rate_by_axis": dict(t.get("learn") or {}),
                        "bias_offset_by_axis": dict(t.get("bias") or {}),
                        "style_tags": list(t.get("style_tags") or []),
                        "rng_seed": _stable_seed("scout", team_id, specialty_key),
                    }

                    traits = {
                        "experience_years": 0,
                        "reputation": "avg",
                    }

                    cur.execute(
                        """
                        INSERT INTO scouting_scouts(
                            scout_id, team_id, display_name, specialty_key,
                            profile_json, traits_json, is_active,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            scout_id,
                            team_id,
                            display_name,
                            specialty_key,
                            _json_dumps(profile),
                            _json_dumps(traits),
                            1,
                            now,
                            now,
                        ),
                    )
                    created += 1

    return {
        "ok": True,
        "teams": len(teams),
        "scouts_per_team": int(scouts_per_team),
        "created": int(created),
        "existing": int(existing),
    }


# -----------------------------------------------------------------------------
# Kalman update core
# -----------------------------------------------------------------------------


def _kalman_update(*, mu: float, sigma: float, z: float, meas_sigma: float, sigma_floor: float) -> Tuple[float, float]:
    """One-step scalar Kalman update.

    - mu/sigma: prior mean/std
    - z: measurement
    - meas_sigma: measurement std
    - sigma_floor: lower bound for posterior sigma
    """
    mu0 = float(mu)
    s0 = float(max(1e-6, sigma))
    z0 = float(z)
    r0 = float(max(1e-6, meas_sigma))

    P = s0 * s0
    R = r0 * r0
    denom = P + R
    if denom <= 1e-12:
        return mu0, float(max(sigma_floor, s0))
    K = P / denom
    mu1 = mu0 + K * (z0 - mu0)
    P1 = (1.0 - K) * P
    s1 = math.sqrt(float(max(1e-9, P1)))
    s1 = float(max(float(sigma_floor), s1))
    return float(mu1), float(s1)


def _effective_meas_sigma(
    *,
    axis: AxisDef,
    base_days: int,
    acc_mult: float,
    learn_rate: float,
) -> float:
    """Compute measurement noise from observation window length + scout profile."""
    days = float(max(1, int(base_days)))
    lr = float(max(0.05, float(learn_rate)))
    eff = days * lr
    std = float(axis.base_meas_std) * float(max(0.1, float(acc_mult))) / math.sqrt(float(max(1.0, eff)))
    std = float(max(float(axis.meas_floor), std))
    return std


# -----------------------------------------------------------------------------
# Monthly checkpoint
# -----------------------------------------------------------------------------


def run_monthly_scouting_checkpoints(
    db_path: str,
    *,
    from_date: str,
    to_date: str,
    min_days_assigned_for_report: int = 14,
) -> Dict[str, Any]:
    """Run end-of-month scouting report checkpoints between two dates.

    Only months whose month_end <= to_date are processed.

    Idempotent:
      - scouting_reports has UNIQUE(assignment_id, period_key)
      - assignment progress is only updated when a report row is created
    """
    d0 = _parse_date_iso(from_date, field="from_date")
    d1 = _parse_date_iso(to_date, field="to_date")
    if d1 < d0:
        d0, d1 = d1, d0

    min_days = int(min_days_assigned_for_report)
    if min_days < 0:
        min_days = 0

    # Pre-compute month periods to process.
    periods: List[Tuple[str, _dt.date, _dt.date]] = []  # (period_key, month_start, month_end)
    cur = _month_floor(d0)
    end_floor = _month_floor(d1)
    while cur <= end_floor:
        pk = f"{cur.year:04d}-{cur.month:02d}"
        mend = _month_end(cur)
        # Only process if month end is within [d0..d1].
        if mend < d0:
            cur = _add_one_month(cur)
            continue
        if mend > d1:
            # This month hasn't ended yet (relative to to_date). Stop.
            break
        periods.append((pk, cur, mend))
        cur = _add_one_month(cur)

    # Fast no-op: if there are no periods, nothing to do.
    if not periods:
        return {
            "ok": True,
            "from_date": str(from_date),
            "to_date": str(to_date),
            "handled": [],
            "generated_reports": 0,
            "skipped": {"no_periods": True},
        }

    now = game_time.now_utc_like_iso()
    handled: List[Dict[str, Any]] = []
    total_created = 0

    with LeagueRepo(db_path) as repo:
        repo.init_db()

        # Global no-op if there are no ACTIVE assignments at all.
        any_active = repo._conn.execute(
            "SELECT 1 FROM scouting_assignments WHERE status='ACTIVE' LIMIT 1;"
        ).fetchone()
        if not any_active:
            return {
                "ok": True,
                "from_date": str(from_date),
                "to_date": str(to_date),
                "handled": [],
                "generated_reports": 0,
                "skipped": {"no_active_assignments": True},
            }

        for pk, mstart, mend in periods:
            as_of = mend.isoformat()
            season_year = int(_infer_college_season_year_from_date(mend))

            created = 0
            skipped_recent = 0
            skipped_existing = 0
            skipped_missing_player = 0
            skipped_not_assigned_yet = 0
            skipped_no_days = 0

            with repo.transaction() as cur:
                # Load all ACTIVE assignments (user-driven) that could apply to this month end.
                # We filter by assigned_date <= as_of and (ended_date is null or ended_date >= as_of).
                rows = cur.execute(
                    """
                    SELECT assignment_id, team_id, scout_id, target_player_id, target_kind,
                           assigned_date, ended_date, progress_json
                    FROM scouting_assignments
                    WHERE status='ACTIVE';
                    """
                ).fetchall()

                for r in rows:
                    assignment_id = str(r[0])
                    team_id = str(r[1])
                    scout_id = str(r[2])
                    player_id = str(r[3])
                    target_kind = str(r[4] or "COLLEGE")
                    assigned_date_s = str(r[5] or "")[:10]
                    ended_date_s = str(r[6] or "")[:10] if r[6] else ""
                    progress = _json_loads(r[7], default={})
                    if not isinstance(progress, dict):
                        progress = {}

                    # Date gating
                    try:
                        assigned_d = _dt.date.fromisoformat(assigned_date_s)
                    except Exception:
                        # Malformed assignment row; skip safely.
                        skipped_not_assigned_yet += 1
                        continue

                    if assigned_d > mend:
                        skipped_not_assigned_yet += 1
                        continue
                    if ended_date_s:
                        try:
                            ended_d = _dt.date.fromisoformat(ended_date_s)
                        except Exception:
                            ended_d = None
                        if ended_d is not None and ended_d < mend:
                            # Not active at this month end.
                            continue

                    days_since_assigned = int((mend - assigned_d).days)
                    if days_since_assigned <= min_days:
                        skipped_recent += 1
                        continue

                    # Idempotency: if report already exists for this assignment+month, skip.
                    exists = cur.execute(
                        """
                        SELECT 1 FROM scouting_reports
                        WHERE assignment_id=? AND period_key=?
                        LIMIT 1;
                        """,
                        (assignment_id, pk),
                    ).fetchone()
                    if exists:
                        skipped_existing += 1
                        continue

                    # Load scout profile
                    srow = cur.execute(
                        """
                        SELECT display_name, specialty_key, profile_json
                        FROM scouting_scouts
                        WHERE scout_id=?
                        LIMIT 1;
                        """,
                        (scout_id,),
                    ).fetchone()
                    if not srow:
                        # Staff row missing (should not happen if seeded)
                        continue
                    scout_name = str(srow[0] or "Scout")
                    specialty_key = str(srow[1] or "GENERAL")
                    scout_profile = _json_loads(srow[2], default={})
                    if not isinstance(scout_profile, dict):
                        scout_profile = {}

                    focus_axes = scout_profile.get("focus_axes")
                    if not isinstance(focus_axes, list) or not focus_axes:
                        focus_axes = []

                    acc_mult_by_axis = scout_profile.get("acc_mult_by_axis")
                    if not isinstance(acc_mult_by_axis, dict):
                        acc_mult_by_axis = {}
                    learn_rate_by_axis = scout_profile.get("learn_rate_by_axis")
                    if not isinstance(learn_rate_by_axis, dict):
                        learn_rate_by_axis = {}
                    bias_offset_by_axis = scout_profile.get("bias_offset_by_axis")
                    if not isinstance(bias_offset_by_axis, dict):
                        bias_offset_by_axis = {}
                    style_tags = scout_profile.get("style_tags")
                    if not isinstance(style_tags, list):
                        style_tags = []

                    # Load target player (college only for now)
                    prow = None
                    if str(target_kind).upper() == "COLLEGE":
                        prow = cur.execute(
                            """
                            SELECT player_id, name, pos, age, height_in, weight_lb,
                                   college_team_id, class_year, entry_season_year, status,
                                   ovr, attrs_json
                            FROM college_players
                            WHERE player_id=?
                            LIMIT 1;
                            """,
                            (player_id,),
                        ).fetchone()

                    if not prow:
                        skipped_missing_player += 1
                        continue

                    # Unpack player
                    p_name = str(prow[1] or "Unknown")
                    p_pos = str(prow[2] or "G")
                    p_age = int(prow[3] or 19)
                    p_h = int(prow[4] or 78)
                    p_w = int(prow[5] or 210)
                    p_team = str(prow[6] or "")
                    p_class = int(prow[7] or 1)
                    p_entry_sy = int(prow[8] or season_year)
                    p_status = str(prow[9] or "ACTIVE")
                    p_ovr = int(prow[10] or 60)
                    p_attrs = _json_loads(prow[11], default={})
                    if not isinstance(p_attrs, dict):
                        p_attrs = {}

                    player_snapshot = {
                        "player_id": str(player_id),
                        "name": p_name,
                        "pos": p_pos,
                        "age": int(p_age),
                        "height_in": int(p_h),
                        "weight_lb": int(p_w),
                        "college_team_id": p_team,
                        "class_year": int(p_class),
                        "entry_season_year": int(p_entry_sy),
                        "status": p_status,
                    }

                    # Determine observation window length since last report (or assignment).
                    last_obs_s = None
                    if isinstance(progress.get("last_obs_date"), str):
                        last_obs_s = str(progress.get("last_obs_date"))[:10]
                    last_obs_d: Optional[_dt.date] = None
                    if last_obs_s:
                        try:
                            last_obs_d = _dt.date.fromisoformat(last_obs_s)
                        except Exception:
                            last_obs_d = None

                    # Window start: max(assigned_date, last_obs_date + 1)
                    if last_obs_d is not None:
                        window_start = max(assigned_d, last_obs_d + _dt.timedelta(days=1))
                    else:
                        window_start = assigned_d
                    window_end = mend
                    if window_start > window_end:
                        skipped_no_days += 1
                        continue
                    days_covered = int((window_end - window_start).days) + 1
                    if days_covered <= 0:
                        skipped_no_days += 1
                        continue

                    # Compute true axes
                    true_axes = _compute_true_axis_values(ovr=p_ovr, attrs=p_attrs)

                    # Prepare/update progress axes state
                    axes_state = progress.get("axes")
                    if not isinstance(axes_state, dict):
                        axes_state = {}

                    updated_axes: Dict[str, Any] = {}
                    report_estimates: Dict[str, Any] = {}

                    # Which axes this scout reports on (focus only).
                    # NOTE: you can expand later to include "secondary" axes.
                    axes_to_update: List[str] = []
                    for ax in focus_axes:
                        k = str(ax)
                        if k in AXES and k not in axes_to_update:
                            axes_to_update.append(k)

                    # As a small baseline, analytics scouts (and only them) also report "overall" and "upside".
                    # Other scouts may have focus without overall.
                    if str(specialty_key).upper() == "ANALYTICS":
                        for k in ("overall", "upside"):
                            if k in AXES and k not in axes_to_update:
                                axes_to_update.append(k)

                    # If still empty, default to overall.
                    if not axes_to_update:
                        axes_to_update = ["overall"]

                    for axis_key in axes_to_update:
                        axis_def = AXES.get(axis_key)
                        if not axis_def:
                            continue

                        st = axes_state.get(axis_key)
                        if not isinstance(st, dict):
                            st = {}
                        mu0 = _safe_float(st.get("mu"), 50.0)
                        sigma0 = _safe_float(st.get("sigma"), axis_def.init_sigma)

                        true_v = _safe_float(true_axes.get(axis_key), 50.0)
                        bias = _safe_float(bias_offset_by_axis.get(axis_key), 0.0)

                        acc_mult = _safe_float(acc_mult_by_axis.get(axis_key), 1.0)
                        learn = _safe_float(learn_rate_by_axis.get(axis_key), 1.0)

                        meas_sigma = _effective_meas_sigma(
                            axis=axis_def,
                            base_days=days_covered,
                            acc_mult=acc_mult,
                            learn_rate=learn,
                        )

                        # Deterministic observation noise (seeded per axis / month / scout / player)
                        rng = random.Random(_stable_seed("scout_obs", scout_id, player_id, pk, axis_key))
                        z = true_v + bias + rng.gauss(0.0, meas_sigma)
                        z = _clamp100(z)

                        mu1, sigma1 = _kalman_update(
                            mu=mu0,
                            sigma=sigma0,
                            z=z,
                            meas_sigma=meas_sigma,
                            sigma_floor=axis_def.sigma_floor,
                        )

                        updated_axes[axis_key] = {
                            "mu": float(round(mu1, 2)),
                            "sigma": float(round(sigma1, 2)),
                            "last_meas_sigma": float(round(meas_sigma, 2)),
                        }
                        report_estimates[axis_key] = {
                            "label": axis_def.label,
                            "value_est": float(round(mu1, 1)),
                            "sigma": float(round(sigma1, 1)),
                            "confidence": _confidence_from_sigma(sigma1),
                            "grade": _grade_from_0_100(mu1),
                            "range_2sigma": [
                                float(round(_clamp100(mu1 - 2.0 * sigma1), 1)),
                                float(round(_clamp100(mu1 + 2.0 * sigma1), 1)),
                            ],
                        }

                    # Update progress JSON
                    axes_state.update(updated_axes)
                    progress["axes"] = axes_state
                    progress["last_obs_date"] = as_of
                    progress["total_obs_days"] = int(_safe_float(progress.get("total_obs_days"), 0.0) + days_covered)
                    progress["updated_at"] = now

                    # Build structured payload for LLM (text generation is done elsewhere)
                    payload = {
                        "schema_version": 1,
                        "method": "kalman_v1",
                        "period_key": pk,
                        "as_of_date": as_of,
                        "days_covered": int(days_covered),
                        "scout": {
                            "scout_id": scout_id,
                            "display_name": scout_name,
                            "specialty_key": specialty_key,
                            "style_tags": style_tags,
                            "focus_axes": list(axes_to_update),
                        },
                        "player": {
                            "player_id": str(player_id),
                            "name": p_name,
                            "pos": p_pos,
                            "age": int(p_age),
                            "height_in": int(p_h),
                            "weight_lb": int(p_w),
                            "college_team_id": p_team,
                            "class_year": int(p_class),
                            "entry_season_year": int(p_entry_sy),
                            "status": p_status,
                        },
                        "estimates": report_estimates,
                        "meta": {
                            "bias_note": "Scout reports include systematic biases; confidence indicates uncertainty, not correctness.",
                        },
                    }

                    report_id = f"SREP_{assignment_id}_{pk}"

                    cur.execute(
                        """
                        INSERT INTO scouting_reports(
                            report_id, assignment_id, team_id, scout_id,
                            target_player_id, target_kind,
                            season_year, period_key, as_of_date,
                            days_covered, player_snapshot_json,
                            payload_json, report_text, status, llm_meta_json,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            report_id,
                            assignment_id,
                            team_id,
                            scout_id,
                            player_id,
                            target_kind,
                            int(season_year),
                            pk,
                            as_of,
                            int(days_covered),
                            _json_dumps(player_snapshot),
                            _json_dumps(payload),
                            None,  # report_text (LLM-generated elsewhere / on-demand)
                            "READY_STRUCT",
                            _json_dumps({"note": "text_generation_deferred"}),
                            now,
                            now,
                        ),
                    )

                    # Persist assignment progress
                    cur.execute(
                        """
                        UPDATE scouting_assignments
                        SET progress_json=?, updated_at=?
                        WHERE assignment_id=?;
                        """,
                        (_json_dumps(progress), now, assignment_id),
                    )

                    created += 1

            handled.append(
                {
                    "period_key": pk,
                    "as_of_date": as_of,
                    "season_year": int(season_year),
                    "created": int(created),
                    "skipped": {
                        "recent_assignment": int(skipped_recent),
                        "existing": int(skipped_existing),
                        "missing_player": int(skipped_missing_player),
                        "not_assigned_yet": int(skipped_not_assigned_yet),
                        "no_days": int(skipped_no_days),
                    },
                }
            )
            total_created += int(created)

    return {
        "ok": True,
        "from_date": str(from_date),
        "to_date": str(to_date),
        "handled": handled,
        "generated_reports": int(total_created),
    }
