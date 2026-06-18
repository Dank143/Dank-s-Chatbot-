import asyncio
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


_CREATOR_KEYWORDS_EN = {"creator", "maker", "developer", "father", "daddy"}
_CREATOR_RESPONSE_EN = "ALL HAIL MISTER DANG! WOOOOOO BABIIIIII!"

_CREATOR_PHRASES_VI = ["tạo ra", "làm ra", "code ra", "thiết kế", "nhà sáng tạo", "cha đẻ", "ông trùm"]
_CREATOR_RESPONSE_VI = "QUÝ NGÀI ĐĂNG VĨ ĐẠI. SIUUUUUU!"


def is_asking_about_creator(text: str) -> str | None:
    """Canned hail response if the text asks who made the bot, else None."""
    if any(p in text.lower() for p in _CREATOR_PHRASES_VI):
        return _CREATOR_RESPONSE_VI
    pattern = r'\byour\s+(?:' + '|'.join(_CREATOR_KEYWORDS_EN) + r')\b'
    if re.search(pattern, text, re.IGNORECASE):
        return _CREATOR_RESPONSE_EN
    return None


# Non-Latin script detector (CJK, Hangul, Arabic, Cyrillic).
_NON_LATIN_RE = re.compile(
    r'[぀-ヿ㐀-䶿一-鿿가-힯'
    r'豈-﫿؀-ۿЀ-ӿ]'
)


def is_non_english(text: str, threshold: float = 0.15) -> bool:
    if not text:
        return False
    return len(_NON_LATIN_RE.findall(text)) / len(text) > threshold


def parse_attachments(raw: str | None) -> dict:
    """Parse attachments JSON into {images, documents}; legacy list → images."""
    if not raw:
        return {"images": [], "documents": []}
    data = json.loads(raw)
    if isinstance(data, list):
        return {"images": data, "documents": []}
    return {"images": data.get("images", []), "documents": data.get("documents", [])}


def build_api_content(text: str, attachments: dict) -> "str | list":
    """Build plain text or multimodal content parts for a message."""
    images = attachments.get("images", [])
    documents = attachments.get("documents", [])

    full_text = text
    if documents:
        doc_blocks = "\n\n".join(
            f"[Attached file: {d['name']}]\n{d['text']}" for d in documents
        )
        full_text = doc_blocks + ("\n\n" + text if text else "")

    if not images:
        return full_text
    parts: list[dict] = [{"type": "text", "text": full_text}] if full_text else []
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": img}})
    return parts


def _to_parts(content: "str | list") -> list[dict]:
    return content if isinstance(content, list) else [{"type": "text", "text": content}]



def reasoning_controls(reasoning: str | None) -> tuple[dict, str | None]:
    """Map a model's YAML `reasoning` tag to (extra create() kwargs, system suffix).

    Currently disabled — thinking runs freely. Uncomment the block below
    to re-enable per-family suppression (qwen, gptoss, nemotron, etc.).
    """
    # if reasoning == "qwen":
    #     return {"extra_body": {"chat_template_kwargs": {"thinking": False}}}, None
    # if reasoning == "gptoss":
    #     return {"reasoning_effort": "low"}, None
    # if reasoning == "nemotron":
    #     return {}, "detailed thinking off"
    # if reasoning == "minimax":
    #     return {"extra_body": {"thinking": {"type": "disabled"}}}, None
    # if reasoning == "kimi":
    #     return {}, "You must respond directly. Do not use extended thinking or output reasoning chains."
    # if reasoning == "glm":
    #     return {}, "You must respond directly. Do not use extended thinking or output reasoning chains."
    return {}, None


def build_messages(history, system_prompt: str | None, today: str | None = None,
                   extra_system: str | None = None) -> list[dict]:
    """Convert DB history rows into OpenAI-format messages list."""
    rows = list(history)
    # Drop leading assistant rows (truncation may start mid-exchange)
    while rows and rows[0]["role"] != "user":
        rows = rows[1:]

    last_user_idx = max(
        (i for i, r in enumerate(rows) if r["role"] == "user"),
        default=None,
    )
    raw: list[dict] = []
    for i, r in enumerate(rows):
        att = parse_attachments(r["attachments"])
        had_images = bool(att["images"])
        had_documents = bool(att["documents"])
        # Keep attachments only on the last user turn to limit prompt size.
        if i != last_user_idx:
            att["images"] = []
            att["documents"] = []
        content = build_api_content(r["content"], att)
        # Substitute placeholder for empty content (stripped attachment-only turns).
        if isinstance(content, str) and not content.strip():
            content = "[image]" if had_images else "[document]" if had_documents else "[no content]"
        raw.append({"role": r["role"], "content": content})

    # merge consecutive same-role messages
    messages: list[dict] = []
    for msg in raw:
        if messages and messages[-1]["role"] == msg["role"]:
            prev, curr = messages[-1]["content"], msg["content"]
            if isinstance(prev, str) and isinstance(curr, str):
                messages[-1]["content"] = prev + "\n\n" + curr
            else:
                messages[-1]["content"] = _to_parts(prev) + _to_parts(curr)
        else:
            messages.append(msg)

    sys_parts: list[str] = []
    if today:
        sys_parts.append(f"Today's date is {today}.")
    if extra_system:
        sys_parts.append(extra_system)
    if system_prompt:
        sys_parts.append(system_prompt)
    if sys_parts:
        messages = [{"role": "system", "content": "\n".join(sys_parts)}] + messages
    return messages


_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"
_CHANNEL_OPEN_RE = re.compile(r'<\|channel>[^\n]*\n?')


async def llm_stream(client, model, messages, max_tokens, temperature, result: dict,
                     extra_create: dict | None = None):
    """Stream model reply as SSE deltas, filtering <think> tags."""
    full_content: list[str] = []
    finish_reason = None
    lang_checked = False
    in_think = False
    tag_buf = ""  # holds tail chars that might be a partial tag
    raw_chunks: list[str] = []  # raw model text (pre-think-filter), for blank rescue
    t_start = time.monotonic()  # request sent
    t_first = None              # first content chunk received (time-to-first-token)

    def _flush(text: str) -> str:
        """Filter text through think-tag state machine."""
        nonlocal in_think, tag_buf
        s = tag_buf + text
        tag_buf = ""
        out = []
        while s:
            if in_think:
                idx = s.find(_THINK_CLOSE)
                if idx == -1:
                    tag_buf = s[-(len(_THINK_CLOSE) - 1):]
                    break
                s = s[idx + len(_THINK_CLOSE):].lstrip("\n")
                in_think = False
            else:
                idx = s.find(_THINK_OPEN)
                if idx == -1:
                    keep = len(_THINK_OPEN) - 1
                    out.append(s[:-keep] if len(s) > keep else "")
                    tag_buf = s[-keep:] if len(s) >= keep else s
                    break
                out.append(s[:idx])
                s = s[idx + len(_THINK_OPEN):]
                in_think = True
        return "".join(out)

    last_exc = None
    for attempt in range(2):
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                **(extra_create or {}),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if not delta or not delta.content:
                    continue
                if t_first is None:
                    t_first = time.monotonic()
                chunk = _CHANNEL_OPEN_RE.sub(_THINK_OPEN, delta.content).replace("<channel|>", _THINK_CLOSE)
                raw_chunks.append(chunk)
                filtered = _flush(chunk)
                if not filtered:
                    continue
                full_content.append(filtered)
                if not lang_checked:
                    sample = "".join(full_content)
                    # Wait for 60+ chars to avoid false positives from stray CJK.
                    if len(sample) >= 60:
                        lang_checked = True
                        if is_non_english(sample, threshold=0.30):
                            logger.warning("llm_abort non_english model=%s sample=%r", model, sample[:60])
                            yield f"data: {json.dumps({'type': 'error', 'message': 'Model responded in a non-English language. Try rephrasing your question or switching models.'})}\n\n"
                            return
                yield f"data: {json.dumps({'type': 'delta', 'content': filtered})}\n\n"
            last_exc = None
            break
        except Exception as exc:
            # Don't retry rate-limits (429).
            is_rate_limit = "429" in str(exc) or getattr(exc, "status_code", None) == 429
            if is_rate_limit:
                last_exc = exc
                break
            # Retry once if nothing streamed yet (cold-start malformed chunk).
            if attempt == 0 and not full_content:
                logger.warning("llm_stream early-error retry model=%s: %s", model, exc)
                finish_reason = None
                in_think = False
                tag_buf = ""
                lang_checked = False
                t_first = None
                raw_chunks.clear()
                await asyncio.sleep(0.6)
                continue
            last_exc = exc
            break

    if last_exc is not None:
        raw_len = sum(len(c) for c in raw_chunks)
        logger.warning("llm_stream error model=%s finish=%s raw=%d: %s", model, finish_reason, raw_len, last_exc)
        is_rate_limit = "429" in str(last_exc) or getattr(last_exc, "status_code", None) == 429
        err_msg = "Rate limited by NIM — too many requests. Wait a moment then try again, or switch models." if is_rate_limit else str(last_exc)
        yield f"data: {json.dumps({'type': 'error', 'message': err_msg})}\n\n"
        return

    # flush any buffered tail after stream ends
    if not in_think and tag_buf.strip():
        full_content.append(tag_buf)
        yield f"data: {json.dumps({'type': 'delta', 'content': tag_buf})}\n\n"

    # Rescue unclosed <think>: replay raw stream if visible output is blank.
    if in_think and not "".join(full_content).strip():
        salvaged = "".join(raw_chunks).replace(_THINK_OPEN, "").strip()
        if salvaged:
            logger.warning("llm_rescue unclosed_think model=%s chars=%d", model, len(salvaged))
            full_content.append(salvaged)
            yield f"data: {json.dumps({'type': 'delta', 'content': salvaged})}\n\n"

    visible = "".join(full_content)
    raw_len = sum(len(c) for c in raw_chunks)
    now = time.monotonic()
    ttft = (t_first - t_start) if t_first else (now - t_start)
    gen = (now - t_first) if t_first else 0.0
    # Diagnostic log: finish=length→token cap, raw>0 visible=0→think-eaten.
    log = logger.warning if (not visible.strip() or finish_reason not in ("stop", None)) else logger.info
    log(
        "llm_done model=%s finish=%s raw_chars=%d visible_chars=%d unclosed_think=%s "
        "ttft=%.2fs gen=%.2fs total=%.2fs",
        model, finish_reason, raw_len, len(visible), in_think,
        ttft, gen, now - t_start,
    )

    # Blank reply left as "" — caller decides retry/surface behavior.
    result["content"] = visible
    result["finish_reason"] = finish_reason
