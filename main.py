import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure our loggers reach the console (uvicorn only configures its own).
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from database import init_db
from routers.chats import router as chats_router
from routers.files import router as files_router
from routers.messages import router as messages_router
from routers.settings import router as settings_router, _ping_model
from search import warmup as _warmup_search

# Aux model for title generation and search-query rewriting; warmed on boot.
_AUX_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm aux LLM and DDG session concurrently at startup.
    asyncio.create_task(_ping_model(_AUX_MODEL, "nim"))
    asyncio.create_task(_warmup_search())
    yield


app = FastAPI(title="NIM Chatbot", lifespan=lifespan)

_BOOT_ID = str(uuid.uuid4())

@app.get("/api/boot-id")
def boot_id():
    return {"id": _BOOT_ID}

init_db()

app.include_router(chats_router)
app.include_router(messages_router)
app.include_router(settings_router)
app.include_router(files_router)

icon_dir = Path(__file__).parent / "icon"
if icon_dir.exists():
    app.mount("/icon", StaticFiles(directory=str(icon_dir)), name="icons")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
