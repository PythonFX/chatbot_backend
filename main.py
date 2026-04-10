import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from routers import conversations, chat, files

load_dotenv()

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


@app.get("/health")
async def health():
    return {"status": "ok"}


class ModelSwitchRequest(BaseModel):
    model: str


# In-memory model state
_current_model = "minimax-m2.7"  # default


@app.post("/model/switch")
async def switch_model(req: ModelSwitchRequest):
    global _current_model
    print(f"[Model] Switched to: {req.model} (was: {_current_model})")
    _current_model = req.model
    return {"status": "ok", "model": req.model}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
