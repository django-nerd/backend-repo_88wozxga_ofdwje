import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI(title="PC Tools API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Models
# ----------------------------
class Part(BaseModel):
    type: str = Field(..., description="cpu|gpu|ram|storage|motherboard|psu|case|cooler|monitor|other")
    name: str
    tdp: Optional[int] = Field(None, description="Approx watt draw for the part (GPU/CPU primary)")
    details: Optional[Dict[str, Any]] = None

class BuildInput(BaseModel):
    budget: Optional[int] = Field(None, description="Budget in USD")
    target_resolution: Optional[str] = Field("1080p", description="1080p|1440p|4k")
    target_fps: Optional[int] = Field(60)
    use_case: Optional[str] = Field("gaming", description="gaming|workstation|mixed")
    parts: List[Part] = Field(default_factory=list)

class PSUCalcRequest(BaseModel):
    parts: List[Part] = Field(default_factory=list)
    overclocking: bool = False
    peripherals_watts: int = 30
    headroom_percent: int = 30

class BottleneckRequest(BaseModel):
    cpu: str
    gpu: str
    memory_gb: int = 16
    resolution: str = "1080p"

class CompatRequest(BaseModel):
    parts: List[Part] = Field(default_factory=list)

class FPSRequest(BaseModel):
    cpu_class: str = Field("mid", description="low|mid|high|ultra")
    gpu_class: str = Field("mid", description="low|mid|high|ultra")
    resolution: str = Field("1080p", description="1080p|1440p|4k")
    game_profile: str = Field("esports", description="esports|aaa|rtx")

class AIOptimizerRequest(BuildInput):
    pass


# ----------------------------
# Utility functions
# ----------------------------
RESOLUTION_MULTIPLIER = {"1080p": 1.0, "1440p": 0.72, "4k": 0.45}
GPU_CLASS_BASE_FPS = {"low": 90, "mid": 140, "high": 200, "ultra": 260}
CPU_CLASS_PENALTY = {"low": 0.75, "mid": 0.9, "high": 1.0, "ultra": 1.05}
GAME_PROFILE_PENALTY = {"esports": 1.0, "aaa": 0.8, "rtx": 0.6}


def estimate_psu_wattage(parts: List[Part], overclocking: bool, peripherals_watts: int, headroom_percent: int) -> Dict[str, Any]:
    cpu_tdp = max([p.tdp or 0 for p in parts if p.type.lower() == "cpu"] + [65])
    gpu_tdp = max([p.tdp or 0 for p in parts if p.type.lower() == "gpu"] + [120])
    other_draw = sum([p.tdp or 0 for p in parts if p.type.lower() not in ("cpu", "gpu")])

    base = cpu_tdp + gpu_tdp + other_draw + peripherals_watts
    if overclocking:
        base *= 1.15
    recommended = int(base * (1 + headroom_percent / 100))

    # Round to common PSU sizes
    std_sizes = [450, 550, 650, 750, 850, 1000, 1200, 1600]
    closest = next((s for s in std_sizes if s >= recommended), std_sizes[-1])

    return {
        "cpu_tdp": cpu_tdp,
        "gpu_tdp": gpu_tdp,
        "other_draw": other_draw,
        "peripherals": peripherals_watts,
        "headroom_percent": headroom_percent,
        "estimated_load_w": int(base),
        "recommended_w": closest,
    }


def simple_bottleneck(cpu: str, gpu: str, memory_gb: int, resolution: str) -> Dict[str, Any]:
    cpu_rank = 1 if any(k in cpu.lower() for k in ["celeron", "pentium", "r3", "i3"]) else 2 if any(k in cpu.lower() for k in ["r5", "i5"]) else 3 if any(k in cpu.lower() for k in ["r7", "i7"]) else 4
    gpu_rank = 1 if any(k in gpu.lower() for k in ["gt 10", "rx 5", "intel arc a3"]) else 2 if any(k in gpu.lower() for k in ["gtx", "rx 6", "a5"]) else 3 if any(k in gpu.lower() for k in ["rtx 306", "rx 67", "a7"]) else 4
    mem_ok = memory_gb >= (16 if resolution in ("1440p", "4k") else 12)
    diff = gpu_rank - cpu_rank
    status = "balanced"
    if diff >= 2:
        status = "cpu_bottleneck"
    elif diff <= -2:
        status = "gpu_bottleneck"
    note = "Memory sufficient" if mem_ok else "Consider upgrading RAM"
    return {"cpu_rank": cpu_rank, "gpu_rank": gpu_rank, "status": status, "note": note}


def estimate_fps(cpu_class: str, gpu_class: str, resolution: str, game_profile: str) -> Dict[str, Any]:
    base = GPU_CLASS_BASE_FPS.get(gpu_class, 120)
    base *= CPU_CLASS_PENALTY.get(cpu_class, 0.9)
    base *= RESOLUTION_MULTIPLIER.get(resolution, 1.0)
    base *= GAME_PROFILE_PENALTY.get(game_profile, 0.8)
    return {
        "estimated_fps": int(base),
        "cpu_class": cpu_class,
        "gpu_class": gpu_class,
        "resolution": resolution,
        "game_profile": game_profile,
    }


def gemini_optimize(prompt: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    # Use Gemini via REST (Generative Language API - text models)
    # We keep this generic so it works with a standard API key environment variable
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    try:
        r = requests.post(url, params={"key": api_key}, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Extract text safely
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )
        return text
    except Exception as e:
        print("Gemini API error:", e)
        return None


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "PC Tools Backend is running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.post("/api/tools/psu")
def psu_calculator(req: PSUCalcRequest):
    return estimate_psu_wattage(req.parts, req.overclocking, req.peripherals_watts, req.headroom_percent)


@app.post("/api/tools/bottleneck")
def bottleneck(req: BottleneckRequest):
    return simple_bottleneck(req.cpu, req.gpu, req.memory_gb, req.resolution)


@app.post("/api/tools/compatibility")
def compatibility(req: CompatRequest):
    # Very lightweight checks as a starting point
    types = [p.type.lower() for p in req.parts]
    issues = []
    if types.count("cpu") != 1:
        issues.append("Recommend exactly one CPU.")
    if types.count("gpu") == 0:
        issues.append("No GPU detected for gaming builds.")
    if types.count("motherboard") != 1:
        issues.append("Recommend exactly one motherboard.")
    if sum(1 for t in types if t == "psu") != 1:
        issues.append("Recommend exactly one PSU.")
    ram_modules = [p for p in req.parts if p.type.lower() == "ram"]
    if len(ram_modules) == 0:
        issues.append("At least one RAM module is required.")
    return {"compatible": len(issues) == 0, "issues": issues}


@app.post("/api/tools/fps")
def fps(req: FPSRequest):
    return estimate_fps(req.cpu_class, req.gpu_class, req.resolution, req.game_profile)


@app.post("/api/tools/upgrade")
def upgrade(req: BuildInput):
    # Heuristic suggestions
    suggestions = []
    cpu_tdps = [p.tdp or 0 for p in req.parts if p.type.lower() == "cpu"]
    gpu_tdps = [p.tdp or 0 for p in req.parts if p.type.lower() == "gpu"]
    if cpu_tdps and max(cpu_tdps) < 65:
        suggestions.append("Consider upgrading CPU for better multitasking and 1% lows.")
    if gpu_tdps and max(gpu_tdps) < 160 and req.use_case == "gaming":
        suggestions.append("Upgrade GPU for higher FPS at your target resolution.")
    if req.target_resolution in ("1440p", "4k"):
        suggestions.append("Aim for 32GB RAM for high-resolution gaming.")
    return {"suggestions": suggestions or ["Your build looks balanced. Consider faster storage for snappier loads."]}


@app.post("/api/tools/optimizer")
def ai_optimizer(req: AIOptimizerRequest):
    # Build a structured prompt for Gemini
    parts_summary = "\n".join([f"- {p.type.upper()}: {p.name} (TDP {p.tdp or 'n/a'}W)" for p in req.parts])
    prompt = f"""
You are an expert PC builder and performance tuner.
User context:
- Budget: {req.budget or 'unspecified'} USD
- Target: {req.target_resolution} @ {req.target_fps} FPS
- Use-case: {req.use_case}
- Current parts:\n{parts_summary}

Tasks:
1) Identify bottlenecks and weakest links.
2) Recommend 2-3 upgrade paths within budget (if provided).
3) Suggest PSU wattage and headroom.
4) Provide expected FPS range for esports and AAA titles.
Keep it concise with bullet points.
""".strip()

    ai_text = gemini_optimize(prompt)

    # Also compute deterministic helpers
    psu = estimate_psu_wattage(req.parts, overclocking=False, peripherals_watts=30, headroom_percent=30)

    result = {
        "ai": ai_text or "Set GEMINI_API_KEY to enable AI recommendations.",
        "psu": psu,
    }
    return result


@app.get("/test")
def test_database():
    """Test endpoint to check if database layer is importable and env set"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = getattr(db, 'name', "✅ Connected")
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
