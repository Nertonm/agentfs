"""
context_manager.py — token budget tracking, memory pressure zones, and eviction.

Three zones (fraction of CONTEXT_BUDGET):
  Normal   (< 70%): no action
  Advisory (70–85%): evict low-priority LRU chunks from cache
  Critical (> 85%): compress oldest history half + flush summary to notebook

The context manager does NOT store actual LLM tokens — it tracks an
approximation (1 token ≈ 4 chars, or pass explicit token counts).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ctxvault.config import (
    CACHE_MAX_ITEMS,
    CONTEXT_BUDGET,
    ZONE_ADVISORY_THRESHOLD,
    ZONE_CRITICAL_THRESHOLD,
)

Zone = str  # "normal" | "advisory" | "critical"


def _approx_tokens(text: str) -> int:
    """Rough token approximation: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


@dataclass
class CacheItem:
    item_id: str
    content: str
    token_count: int
    importance: float = 0.5
    pinned: bool = False
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class ContextManager:
    """
    Manages token budget and eviction across three memory-pressure zones.

    Usage::

        cm = ContextManager()
        cm.add("file:foo.py:0", content, importance=0.8)
        zone = cm.zone()                 # "normal" / "advisory" / "critical"
        evicted = cm.maybe_evict()       # list of evicted item IDs
        status = cm.status_line()        # "[CTX: 68% | ZONE: normal | ...]"
    """

    def __init__(
        self,
        budget: int = CONTEXT_BUDGET,
        *,
        on_evict: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._budget = budget
        self._cache: Dict[str, CacheItem] = {}
        self._used_tokens: int = 0
        self._on_evict = on_evict  # callback(item_id, reason)
        # Observability
        self.eviction_log: List[Tuple[float, str, str]] = []  # (ts, id, reason)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def used(self) -> int:
        return self._used_tokens

    @property
    def free(self) -> int:
        return max(0, self._budget - self._used_tokens)

    def fraction_used(self) -> float:
        return self._used_tokens / self._budget if self._budget else 0.0

    def zone(self) -> Zone:
        f = self.fraction_used()
        if f >= ZONE_CRITICAL_THRESHOLD:
            return "critical"
        if f >= ZONE_ADVISORY_THRESHOLD:
            return "advisory"
        return "normal"

    def add(
        self,
        item_id: str,
        content: str,
        *,
        importance: float = 0.5,
        tokens: Optional[int] = None,
        pinned: bool = False,
    ) -> bool:
        """Add or update an item. Returns True if added without eviction."""
        tc = tokens if tokens is not None else _approx_tokens(content)
        if item_id in self._cache:
            old = self._cache[item_id]
            self._used_tokens -= old.token_count
            old.content = content
            old.token_count = tc
            old.importance = importance
            old.touch()
            self._used_tokens += tc
            return True

        item = CacheItem(
            item_id=item_id,
            content=content,
            token_count=tc,
            importance=importance,
            pinned=pinned,
        )
        self._cache[item_id] = item
        self._used_tokens += tc
        return True

    def remove(self, item_id: str) -> Optional[str]:
        """Remove an item by ID. Returns its content or None."""
        item = self._cache.pop(item_id, None)
        if item:
            self._used_tokens -= item.token_count
            return item.content
        return None

    def pin(self, item_id: str) -> None:
        if item_id in self._cache:
            self._cache[item_id].pinned = True

    def unpin(self, item_id: str) -> None:
        if item_id in self._cache:
            self._cache[item_id].pinned = False

    def touch(self, item_id: str) -> None:
        if item_id in self._cache:
            self._cache[item_id].touch()

    def maybe_evict(self) -> List[str]:
        """Run eviction according to current zone. Returns evicted IDs."""
        zone = self.zone()
        evicted: List[str] = []
        if zone == "normal":
            return evicted

        candidates = self._eviction_candidates()
        if zone == "advisory":
            # Evict until we drop below advisory threshold
            for item in candidates:
                if self.fraction_used() < ZONE_ADVISORY_THRESHOLD:
                    break
                self._evict(item.item_id, "advisory_lru")
                evicted.append(item.item_id)
        elif zone == "critical":
            # Evict aggressively until below critical threshold
            for item in candidates:
                if self.fraction_used() < ZONE_ADVISORY_THRESHOLD:
                    break
                self._evict(item.item_id, "critical_pressure")
                evicted.append(item.item_id)
        return evicted

    def compress_history(self, history: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        Compress oldest half of history.

        Returns (kept, flushed) where ``flushed`` should be written to notebook.
        """
        if len(history) <= 2:
            return history, []
        mid = len(history) // 2
        flushed = history[:mid]
        kept = history[mid:]
        return kept, flushed

    def status_line(
        self,
        *,
        graph_active: bool = True,
        step_count: int = 0,
    ) -> str:
        pct = int(self.fraction_used() * 100)
        zone = self.zone()
        graph_str = "active" if graph_active else "cold"
        return (
            f"[CTX: {pct}% | ZONE: {zone} | "
            f"GRAPH: {graph_str} | STEPS: {step_count}]"
        )

    def get(self, item_id: str) -> Optional[str]:
        item = self._cache.get(item_id)
        if item:
            item.touch()
            return item.content
        return None

    def list_items(self) -> List[CacheItem]:
        return list(self._cache.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _eviction_candidates(self) -> List[CacheItem]:
        """Return non-pinned items sorted by (importance asc, last_access asc)."""
        items = [i for i in self._cache.values() if not i.pinned]
        items.sort(key=lambda x: (x.importance, x.last_access))
        return items

    def _evict(self, item_id: str, reason: str) -> None:
        item = self._cache.pop(item_id, None)
        if item:
            self._used_tokens -= item.token_count
            ts = time.time()
            self.eviction_log.append((ts, item_id, reason))
            if self._on_evict:
                self._on_evict(item_id, reason)
