from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional
from uuid import uuid4

import state
from config import SAVE_ROOT_DIR
from team_utils import ui_cache_rebuild_all

_SAVE_LOCK = RLock()
_SLOT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,63}$")
_SAVE_FORMAT_VERSION = 1


class SaveError(ValueError):
    pass


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_save_root() -> Path:
    root = Path(SAVE_ROOT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_slot_id(slot_id: str) -> str:
    sid = str(slot_id or "").strip()
    if not sid:
        raise SaveError("slot_id is required")
    if not _SLOT_RE.fullmatch(sid):
        raise SaveError("slot_id must match ^[a-zA-Z0-9][a-zA-Z0-9_-]{2,63}$")
    return sid


def _new_slot_id() -> str:
    return f"slot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"


def _slot_dir(slot_id: str) -> Path:
    return _ensure_save_root() / _validate_slot_id(slot_id)


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SaveError(f"source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(dst)
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _meta_from_state(*, slot_id: str, slot_name: str, save_name: Optional[str], note: Optional[str], save_version: int) -> Dict[str, Any]:
    snap = state.export_full_state_snapshot()
    league = snap.get("league") if isinstance(snap, dict) else {}
    if not isinstance(league, dict):
        league = {}
    return {
        "slot_id": slot_id,
        "slot_name": slot_name,
        "save_name": save_name,
        "note": note,
        "save_version": int(save_version),
        "saved_at": _utc_now_iso(),
        "save_format_version": _SAVE_FORMAT_VERSION,
        "active_season_id": snap.get("active_season_id"),
        "turn": snap.get("turn"),
        "season_year": league.get("season_year"),
        "current_date": league.get("current_date"),
    }


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_game(*, slot_id: str, save_name: Optional[str] = None, note: Optional[str] = None) -> Dict[str, Any]:
    with _SAVE_LOCK:
        sid = _validate_slot_id(slot_id)
        db_path = Path(state.get_db_path())

        slot_dir = _slot_dir(sid)
        slot_dir.mkdir(parents=True, exist_ok=True)

        meta_path = slot_dir / "meta.json"
        prev = _read_meta(meta_path)
        slot_name = str(prev.get("slot_name") or sid)
        save_version = int(prev.get("save_version") or 0) + 1

        _atomic_copy_file(db_path, slot_dir / "league.sqlite3")
        save_state = state.export_save_state_snapshot()
        _atomic_write_json(slot_dir / "state.partial.json", save_state)

        meta = _meta_from_state(
            slot_id=sid,
            slot_name=slot_name,
            save_name=save_name,
            note=note,
            save_version=save_version,
        )
        _atomic_write_json(meta_path, meta)

        return {
            "ok": True,
            "slot_id": sid,
            "save_version": save_version,
            "saved_at": meta["saved_at"],
            "active_season_id": meta.get("active_season_id"),
            "current_date": meta.get("current_date"),
            "season_year": meta.get("season_year"),
        }


def create_new_game(
    *,
    slot_name: str,
    slot_id: Optional[str] = None,
    season_year: Optional[int] = None,
    user_team_id: Optional[str] = None,
    overwrite_if_exists: bool = False,
) -> Dict[str, Any]:
    with _SAVE_LOCK:
        name = str(slot_name or "").strip()
        if not name:
            raise SaveError("slot_name is required")

        sid = _validate_slot_id(slot_id) if slot_id else _new_slot_id()
        slot_dir = _slot_dir(sid)
        if slot_dir.exists() and any(slot_dir.iterdir()) and not overwrite_if_exists:
            raise SaveError(f"slot already exists: {sid}")

        slot_dir.mkdir(parents=True, exist_ok=True)
        new_db_path = slot_dir / "league.sqlite3"

        if new_db_path.exists() and overwrite_if_exists:
            new_db_path.unlink()

        state.reset_state_for_dev()
        state.set_db_path(str(new_db_path))
        state.startup_init_state()

        if season_year is not None:
            state.start_new_season(int(season_year), rebuild_schedule=True)

        try:
            ui_cache_rebuild_all()
        except Exception:
            pass

        save_payload = save_game(slot_id=sid, save_name="new_game_init", note="initial save after new game")

        meta_path = slot_dir / "meta.json"
        meta = _read_meta(meta_path)
        meta["slot_name"] = name
        meta["user_team_id"] = user_team_id
        _atomic_write_json(meta_path, meta)

        out = dict(save_payload)
        out.update(
            {
                "slot_id": sid,
                "slot_name": name,
                "db_path": str(new_db_path),
                "user_team_id": user_team_id,
                "created_at": meta.get("saved_at"),
            }
        )
        return out
