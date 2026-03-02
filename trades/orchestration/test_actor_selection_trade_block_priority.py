import random
import sqlite3
import unittest
from datetime import date
from types import SimpleNamespace

from trades.orchestration.actor_selection import select_trade_actors
from trades.orchestration.market_state import upsert_trade_listing
from trades.orchestration.types import OrchestrationConfig


class ActorSelectionTradeBlockPriorityTests(unittest.TestCase):
    def _base_tick_ctx(self):
        team_situations = {
            "AAA": SimpleNamespace(constraints=SimpleNamespace(cooldown_active=False)),
            "BBB": SimpleNamespace(constraints=SimpleNamespace(cooldown_active=False)),
        }
        return SimpleNamespace(team_situations=team_situations, repo=SimpleNamespace(_conn=None))

    def _config(self):
        return OrchestrationConfig(
            min_active_teams=1,
            max_active_teams=1,
            enable_market_day_rhythm=False,
            enable_dynamic_per_team_max_results=False,
            enable_threads=False,
            pressure_tier_weight_multiplier_high=1.0,
            pressure_tier_weight_multiplier_rush=1.0,
            trade_block_actor_weight_multiplier=100.0,
            trade_block_public_request_multiplier=2.0,
            public_trade_request_actor_add=1.0,
        )

    def test_listing_team_gets_priority_boost(self):
        tick_ctx = self._base_tick_ctx()
        market = {}
        upsert_trade_listing(
            market,
            today=date(2026, 1, 1),
            player_id="p1",
            team_id="AAA",
            listed_by="USER",
            visibility="PUBLIC",
            priority=0.8,
            reason_code="MANUAL",
        )
        picked = select_trade_actors(
            tick_ctx,
            config=self._config(),
            rng=random.Random(1),
            trade_market=market,
            today=date(2026, 1, 1),
        )
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0].team_id, "AAA")

    def test_public_trade_request_without_listing_still_gets_additive_boost(self):
        tick_ctx = self._base_tick_ctx()

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE player_agency_state (
                player_id TEXT,
                team_id TEXT,
                trade_request_level INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO player_agency_state(player_id, team_id, trade_request_level) VALUES ('p2','BBB',2);"
        )
        conn.commit()
        tick_ctx.repo = SimpleNamespace(_conn=conn)

        picked = select_trade_actors(
            tick_ctx,
            config=self._config(),
            rng=random.Random(1),
            trade_market={},
            today=date(2026, 1, 1),
        )
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0].team_id, "BBB")

        conn.close()


if __name__ == "__main__":
    unittest.main()
