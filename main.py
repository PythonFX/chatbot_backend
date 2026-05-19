import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from routers import conversations, chat, files, group_chat
from services.db_service import init_db

load_dotenv()
init_db()

app = FastAPI(title="Chatbot API", version="1.0.0")

_local_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
# Support LAN access: dynamically add any host IP on the 192.168.x.x subnet
_lan_origins = []
for prefix in ("192.168", "10.0", "172.17"):
    try:
        import socket
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip.startswith(prefix):
            _lan_origins.append(f"http://{local_ip}:5173")
            _lan_origins.append(f"http://{local_ip}:3000")
    except Exception:
        pass

all_origins = _local_origins + _lan_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=all_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(group_chat.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


class ModelSwitchRequest(BaseModel):
    model: str


class SettingsRequest(BaseModel):
    thinking_enabled: bool | None = None


@app.get("/settings")
async def get_settings():
    from services.db_service import db_get_all_settings, db_get_setting
    settings = db_get_all_settings()
    return {
        "thinking_enabled": settings.get("thinking_enabled", "true").lower() == "true",
    }


@app.put("/settings")
async def update_settings(req: SettingsRequest):
    from services.db_service import db_set_setting
    if req.thinking_enabled is not None:
        db_set_setting("thinking_enabled", str(req.thinking_enabled).lower())
    return {"status": "ok"}


@app.post("/model/switch")
async def switch_model(req: ModelSwitchRequest):
    from services.llm_manager import set_current_model, get_current_model, get_available_models
    available = get_available_models()
    if req.model not in available:
        return {"status": "error", "error": f"Unknown model: {req.model}", "available": available}
    old = get_current_model()
    set_current_model(req.model)
    return {"status": "ok", "model": req.model, "previous": old}


@app.get("/model/list")
async def list_models():
    from services.llm_manager import get_current_model, get_available_models, get_model_info
    return {
        "current": get_current_model(),
        "available": get_available_models(),
        "models": get_model_info(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8017)
