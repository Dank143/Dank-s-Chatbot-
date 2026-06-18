import asyncio
import uuid

from fastapi import APIRouter, HTTPException

from config import load_config
from database import get_db, now_iso
from schemas import CreateChatBody, UpdateChatBody

router = APIRouter(prefix="/api/chats")


@router.get("")
async def list_chats():
    def _work():
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, title, model, starred, created_at, updated_at "
                "FROM chats ORDER BY starred DESC, updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    return await asyncio.to_thread(_work)


@router.post("", status_code=201)
async def create_chat(body: CreateChatBody):
    cfg = load_config()
    chat_id = str(uuid.uuid4())
    ts = now_iso()
    model = body.model or cfg.get("default_model")

    def _work():
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chats (id, title, model, created_at, updated_at) VALUES (?,?,?,?,?)",
                (chat_id, body.title, model, ts, ts),
            )
    await asyncio.to_thread(_work)
    return {"id": chat_id, "title": body.title, "model": model,
            "starred": 0, "created_at": ts, "updated_at": ts}


@router.get("/{chat_id}")
async def get_chat(chat_id: str):
    def _work():
        with get_db() as conn:
            chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            if not chat:
                raise HTTPException(404, "Chat not found")
            msgs = conn.execute(
                "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)
            ).fetchall()
        return {**dict(chat), "messages": [dict(m) for m in msgs]}
    return await asyncio.to_thread(_work)


@router.patch("/{chat_id}")
async def update_chat(chat_id: str, body: UpdateChatBody):
    def _work():
        with get_db() as conn:
            if not conn.execute("SELECT id FROM chats WHERE id=?", (chat_id,)).fetchone():
                raise HTTPException(404, "Chat not found")
            ts = now_iso()
            sets: list[str] = ["updated_at=?"]
            params: list[str | int] = [ts]
            if body.title is not None:
                sets.append("title=?"); params.append(body.title)
            if body.model is not None:
                sets.append("model=?"); params.append(body.model)
            if body.starred is not None:
                sets.append("starred=?"); params.append(int(body.starred))
            params.append(chat_id)
            conn.execute(f"UPDATE chats SET {', '.join(sets)} WHERE id=?", params)
    await asyncio.to_thread(_work)
    return {"success": True}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    def _work():
        with get_db() as conn:
            conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    await asyncio.to_thread(_work)
    return {"success": True}
