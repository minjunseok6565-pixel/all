import unittest
from types import SimpleNamespace

from trades.valuation.env import ValuationEnv
from trades.valuation.package_effects import PackageEffects, PackageEffectsConfig
from trades.valuation.types import AssetKind, PlayerSnapshot, TeamValuation, ValueComponents


class AgencyDistressValuationLinkTests(unittest.TestCase):
    def _engine(self):
        cfg = PackageEffectsConfig(
            consolidation_scale=0.0,
            roster_excess_waste_rate=0.0,
            hole_penalty_scale=0.0,
            depth_need_scale=0.0,
            cap_flex_scale=0.0,
            cap_room_weight_base=0.0,
            cap_room_value_per_cap_fraction=0.0,
            cap_room_abs_cap=0.0,
            upgrade_scale=0.0,
        )
        return PackageEffects(config=cfg)

    def _ctx(self):
        knobs = SimpleNamespace(
            consolidation_bias=0.5,
            star_premium_exponent=1.0,
            w_now=1.0,
            w_future=1.0,
        )
        policies = SimpleNamespace(fit=SimpleNamespace(need_map={}))
        return SimpleNamespace(team_id="LAL", knobs=knobs, policies=policies, need_map={})

    def _tv(self, pid: str, total: float):
        return TeamValuation(
            asset_key=f"player:{pid}",
            kind=AssetKind.PLAYER,
            ref_id=pid,
            market_value=ValueComponents(now=total, future=0.0),
            team_value=ValueComponents(now=total, future=0.0),
        )

    def _snap(self, pid: str, *, tr: int, tf: float, rf: float):
        return PlayerSnapshot(
            kind="player",
            player_id=pid,
            pos="PG",
            ovr=80,
            team_id="LAL",
            meta={
                "agency_state": {
                    "trade_request_level": tr,
                    "team_frustration": tf,
                    "role_frustration": rf,
                }
            },
        )

    def test_outgoing_distress_increases_seller_acceptability(self):
        eng = self._engine()
        delta, steps, _ = eng.apply(
            team_id="LAL",
            incoming=[],
            outgoing=[(self._tv("p1", 10.0), self._snap("p1", tr=3, tf=0.8, rf=0.7))],
            ctx=self._ctx(),
            env=ValuationEnv.from_trade_rules({}, current_season_year=2026),
        )
        self.assertGreater(delta.total, 0.0)
        self.assertTrue(any(s.code == "AGENCY_DISTRESS_VALUE_ADJUST" for s in steps))

    def test_incoming_distress_decreases_buyer_willingness(self):
        eng = self._engine()
        delta, _, _ = eng.apply(
            team_id="LAL",
            incoming=[(self._tv("p2", 12.0), self._snap("p2", tr=2, tf=0.7, rf=0.4))],
            outgoing=[],
            ctx=self._ctx(),
            env=ValuationEnv.from_trade_rules({}, current_season_year=2026),
        )
        self.assertLess(delta.total, 0.0)


if __name__ == "__main__":
    unittest.main()
