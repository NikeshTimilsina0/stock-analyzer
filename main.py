from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from analysis import analyze, load_csv

BASE = Path(__file__).parent
app = FastAPI(title="Stock Structure Analyzer")


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...), swing: int = 3):
    raw = await file.read()
    try:
        df = load_csv(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return analyze(df, swing)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
