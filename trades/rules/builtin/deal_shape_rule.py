from __future__ import annotations

from dataclasses import dataclass

from ...models import Deal
from ..base import TradeContext


@dataclass
class DealShapeRule:
    rule_id: str = "deal_shape"
    priority: int = 15
    enabled: bool = True

    def validate(self, deal: Deal, ctx: TradeContext) -> None:
        return None
