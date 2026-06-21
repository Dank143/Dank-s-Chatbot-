import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from config import load_config, provider_api, provider_for_model, _PROVIDERS
from database import get_db, now_iso
from search import fetch_web_context, inject_web_context
from llm import build_messages, is_asking_about_creator, llm_stream, reasoning_controls
from schemas import RegenerateBody, SaveAssistantBody, SendMessageBody

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chats")

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

_SEP = r'[\s.\-…_]*'
_DANG_VI_RE = re.compile(r'[Đđ]' + _SEP + r'[Ăă]' + _SEP + r'n' + _SEP + r'g', re.IGNORECASE)
_DANG_EN_RE = re.compile(r'(?<![a-zA-Z])d' + _SEP + r'a' + _SEP + r'n' + _SEP + r'g(?![a-zA-Z])', re.IGNORECASE)
_client_cache: dict[tuple, AsyncOpenAI] = {}

_TITLE_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


def _get_client(api_key: str, base_url: str) -> AsyncOpenAI:
    key = (api_key, base_url)
    if key not in _client_cache:
        _client_cache[key] = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return _client_cache[key]


def _client_for_provider(provider: str) -> AsyncOpenAI:
    """Client for the given provider's current API key/base_url."""
    api = provider_api(provider)
    return _get_client(api["key"], api["base_url"])


def _nim_client() -> AsyncOpenAI:
    """Always return the NIM client (for aux tasks: title gen, etc.)."""
    return _client_for_provider("nim")


def _fallback_title(text: str) -> str:
    return text[:60].strip() + ("…" if len(text) > 60 else "")


def _save_assistant(chat_id: str, msg_id: str, content: str, title: "str | None", model: "str | None" = None) -> None:
    """Persist an assistant message, bump the chat, and optionally set its title."""
    with get_db() as conn:
        ts = now_iso()
        conn.execute(
            "INSERT INTO messages (id, chat_id, role, content, created_at, model) VALUES (?,?,?,?,?,?)",
            (msg_id, chat_id, "assistant", content, ts, model),
        )
        conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id))
        if title:
            conn.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))


def _set_title(chat_id: str, title: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))


def _model_reasoning(model_id: str) -> "str | None":
    """The `reasoning` tag for a model id from models.yaml, or None if untagged."""
    cfg = load_config()
    for p in _PROVIDERS:
        for m in cfg.get(f"models_{p}", []):
            if m.get("id") == model_id:
                return m.get("reasoning")
    return None


def _build_request(history, today: str, model: str):
    """Shared message/generation setup for send + regenerate."""
    defaults = load_config().get("defaults", {})
    max_turns = defaults.get("max_history_turns", 50)
    if len(history) > max_turns:
        history = history[-max_turns:]
    extra_create, extra_system = reasoning_controls(_model_reasoning(model))
    messages = build_messages(history, defaults.get("system_prompt"), today, extra_system)
    prov = provider_for_model(model)
    return (_client_for_provider(prov), messages,
            defaults.get("max_tokens", 2048), defaults.get("temperature", 0.7), extra_create)


async def _generate_title(user_content: str) -> str:
    """LLM-generate a 2-8 word chat title; falls back to truncated text."""
    fallback = _fallback_title(user_content)
    try:
        resp = await _nim_client().chat.completions.create(
            model=_TITLE_MODEL,
            messages=[
                {"role": "system", "content": "Output only a short conversation title: 3-6 words, no punctuation at the end, no quotes, no explanation, no preamble."},
                {"role": "user", "content": user_content[:500]},
            ],
            max_tokens=50,
            temperature=0.0,
            stream=False,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        raw = re.sub(r"^(?:title|name|subject)\s*[:\-–]\s*", "", raw, flags=re.IGNORECASE)
        raw = raw.strip("\"'")
        word_count = len(raw.split())
        if not raw or not (2 <= word_count <= 8):
            return fallback
        return raw
    except Exception as e:
        logger.warning("Title generation failed: %s", e)
        return fallback


@router.post("/{chat_id}/messages")
async def send_message(chat_id: str, body: SendMessageBody):
    cfg = load_config()

    images = body.images or []
    documents = [d.model_dump() for d in (body.documents or [])]
    user_msg_id = str(uuid.uuid4())

    def _setup():
        att_json = json.dumps({"images": images, "documents": documents}) if (images or documents) else None
        with get_db() as conn:
            chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            if not chat:
                raise HTTPException(404, "Chat not found")
            chat = dict(chat)
            model: str = str(body.model or chat["model"] or cfg.get("default_model") or "")
            ts = now_iso()
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, created_at, attachments, model) VALUES (?,?,?,?,?,?,?)",
                (user_msg_id, chat_id, "user", body.content, ts, att_json, model),
            )
            conn.execute("UPDATE chats SET model=?, updated_at=? WHERE id=?", (model, ts, chat_id))
            history = conn.execute(
                "SELECT role, content, attachments FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)
            ).fetchall()
        return model, chat["title"] == "New Chat", history

    model, needs_title, history = await asyncio.to_thread(_setup)

    today = body.client_time or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client, messages, max_tokens, temperature, extra_create = _build_request(history, today, model)

    async def stream_response():
        asst_msg_id = str(uuid.uuid4())
        yield f"data: {json.dumps({'type': 'meta', 'user_msg_id': user_msg_id})}\n\n"

        # Seed auto-title from user text or file names.
        title_seed = body.content
        if not title_seed and documents:
            title_seed = "Uploaded file: " + ", ".join(d["name"] for d in documents)
        title_task = (
            asyncio.create_task(_generate_title(title_seed))
            if needs_title and title_seed else None
        )

        if body.web_search and body.content:
            # Provide recent turns as context for query rewriting.
            ctx_turns = list(history)[:-1][-6:]
            history_context = "\n".join(
                f"{m['role']}: {m['content'][:200]}"
                for m in ctx_turns if m["content"]
            )
            yield f"data: {json.dumps({'type': 'searching', 'query': body.content})}\n\n"
            try:
                web_ctx, search_debug = await asyncio.wait_for(
                    fetch_web_context(body.content, history_context=history_context), timeout=22.0
                )
            except asyncio.TimeoutError:
                web_ctx, search_debug = "", {
                    "site": "general", "original_query": body.content,
                    "rewritten_query": body.content, "query": body.content,
                    "fallback": True, "sources": [], "timed_out": True,
                }
            inject_web_context(messages, web_ctx)
            yield f"data: {json.dumps({'type': 'search_debug', 'got_context': bool(web_ctx), **search_debug})}\n\n"

        creator_resp = is_asking_about_creator(body.content)
        if creator_resp:
            fallback = _fallback_title(body.content) if needs_title and body.content else None
            await asyncio.to_thread(_save_assistant, chat_id, asst_msg_id, creator_resp, fallback, model)
            yield f"data: {json.dumps({'type': 'delta', 'content': creator_resp})}\n\n"
            if fallback:
                yield f"data: {json.dumps({'type': 'title', 'title': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'asst_msg_id': asst_msg_id, 'finish_reason': 'stop'})}\n\n"
            return

        if _DANG_VI_RE.search(body.content):
            hail = "QUÝ NGÀI ĐĂNG VĨ ĐẠI.\n\n"
        elif _DANG_EN_RE.search(body.content):
            hail = "ALL HAIL MISTER DANG!\n\n"
        else:
            hail = ""
        result: dict = {}
        hail_sent = not hail
        # Retry once if the model returns an empty completion on cold start.
        for attempt in range(2):
            result = {}
            async for evt in llm_stream(client, model, messages, max_tokens, temperature, result, extra_create):
                if not hail_sent and '"type": "delta"' in evt:
                    hail_sent = True
                    yield f"data: {json.dumps({'type': 'delta', 'content': hail})}\n\n"
                yield evt
            if "content" not in result:
                return  # stream errored — error event already sent
            if result["content"].strip() or attempt == 1:
                break

        if not result["content"].strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'The model returned an empty response. Please try again.'})}\n\n"
            return

        # Persist answer and emit 'done' before resolving the title.
        await asyncio.to_thread(_save_assistant, chat_id, asst_msg_id, hail + result["content"], None, model)
        yield f"data: {json.dumps({'type': 'done', 'asst_msg_id': asst_msg_id, 'finish_reason': result.get('finish_reason')})}\n\n"

        if needs_title:
            if title_task:
                try:
                    generated = await asyncio.wait_for(title_task, timeout=5.0)
                except Exception:
                    generated = _fallback_title(title_seed)
            else:
                # Image-only turn: no user text to title from — use the reply instead.
                try:
                    generated = await asyncio.wait_for(_generate_title(result["content"]), timeout=5.0)
                except Exception:
                    generated = _fallback_title(result["content"])
            await asyncio.to_thread(_set_title, chat_id, generated)
            yield f"data: {json.dumps({'type': 'title', 'title': generated})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.delete("/{chat_id}/messages/from/{message_id}")
async def delete_messages_from(chat_id: str, message_id: str):
    def _work():
        with get_db() as conn:
            msg = conn.execute(
                "SELECT created_at FROM messages WHERE id=? AND chat_id=?", (message_id, chat_id)
            ).fetchone()
            if not msg:
                raise HTTPException(404, "Message not found")
            conn.execute(
                "DELETE FROM messages WHERE chat_id=? AND created_at >= ?", (chat_id, msg["created_at"])
            )
    await asyncio.to_thread(_work)
    return {"success": True}


@router.post("/{chat_id}/messages/assistant", status_code=201)
async def save_assistant_message(chat_id: str, body: SaveAssistantBody):
    if not body.content.strip():
        raise HTTPException(400, "Empty content")
    msg_id = str(uuid.uuid4())

    def _work():
        with get_db() as conn:
            if not conn.execute("SELECT id FROM chats WHERE id=?", (chat_id,)).fetchone():
                raise HTTPException(404, "Chat not found")
            ts = now_iso()
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, created_at, model) VALUES (?,?,?,?,?,?)",
                (msg_id, chat_id, "assistant", body.content, ts, chat["model"]),
            )
            conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id))
    await asyncio.to_thread(_work)
    return {"id": msg_id}


@router.post("/{chat_id}/regenerate")
async def regenerate_response(chat_id: str, body: RegenerateBody):
    cfg = load_config()

    def _read():
        with get_db() as conn:
            chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
            if not chat:
                raise HTTPException(404, "Chat not found")
            chat = dict(chat)
            model: str = str(body.model or chat["model"] or cfg.get("default_model") or "")
            history = conn.execute(
                "SELECT role, content, attachments FROM messages WHERE chat_id=? ORDER BY created_at", (chat_id,)
            ).fetchall()
        return model, history

    model, history = await asyncio.to_thread(_read)

    if not history:
        raise HTTPException(400, "No messages to regenerate from")

    today = body.client_time or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client, messages_list, max_tokens, temperature, extra_create = _build_request(history, today, model)

    # The message being regenerated is the last user turn; use it as the search query.
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")

    async def stream_response():
        asst_msg_id = str(uuid.uuid4())

        if body.web_search and last_user:
            ctx_turns = [m for m in history if m["content"]][:-1][-6:]
            history_context = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in ctx_turns
            )
            yield f"data: {json.dumps({'type': 'searching', 'query': last_user})}\n\n"
            try:
                web_ctx, search_debug = await asyncio.wait_for(
                    fetch_web_context(last_user, history_context=history_context), timeout=22.0
                )
            except asyncio.TimeoutError:
                web_ctx, search_debug = "", {
                    "site": "general", "original_query": last_user,
                    "rewritten_query": last_user, "query": last_user,
                    "fallback": True, "sources": [], "timed_out": True,
                }
            inject_web_context(messages_list, web_ctx)
            yield f"data: {json.dumps({'type': 'search_debug', 'got_context': bool(web_ctx), **search_debug})}\n\n"

        result: dict = {}
        for attempt in range(2):
            result = {}
            async for evt in llm_stream(client, model, messages_list, max_tokens, temperature, result, extra_create):
                yield evt
            if "content" not in result:
                return  # stream errored — error event already sent
            if result["content"].strip() or attempt == 1:
                break

        if not result["content"].strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'The model returned an empty response. Please try again.'})}\n\n"
            return
        await asyncio.to_thread(_save_assistant, chat_id, asst_msg_id, result["content"], None, model)
        yield f"data: {json.dumps({'type': 'done', 'asst_msg_id': asst_msg_id, 'finish_reason': result.get('finish_reason')})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream", headers=_SSE_HEADERS)
