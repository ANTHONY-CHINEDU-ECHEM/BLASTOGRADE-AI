"""FastAPI inference service for BlastoGrade AI."""
from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from blastograde import __version__
from blastograde.inference.grader import EmbryoGrader, InvalidImageError, get_grader

MAX_BYTES = 15 * 1024 * 1024
MAX_BATCH = 16

app = FastAPI(
    title="BlastoGrade AI",
    version=__version__,
    description="Gardner grading of blastocyst images with per structure Grad CAM heatmaps. "
                "Research prototype. Not a medical device.",
)


def grader_dependency() -> EmbryoGrader:
    try:
        return get_grader()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{upload.filename}: the file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"{upload.filename}: larger than {MAX_BYTES // (1024 * 1024)} MB.")
    return data


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model")
def model_info(grader: EmbryoGrader = Depends(grader_dependency)) -> dict:
    ck = grader.checkpoint
    return {"backbone": ck["backbone"], "image_size": ck["image_size"], "trained_epochs": ck["epoch"],
            "validation_mean_kappa": ck["val_mean_kappa"], "tasks": ck["tasks"], "device": str(grader.device)}


@app.post("/grade")
async def grade(file: UploadFile = File(...), heatmaps: bool = Query(False, description="Return Grad CAM overlays as base64 PNG"),
                grader: EmbryoGrader = Depends(grader_dependency)) -> dict:
    try:
        return {"filename": file.filename, **grader.grade_bytes(await _read(file), heatmaps)}
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/grade/batch")
async def grade_batch(files: list[UploadFile] = File(...), grader: EmbryoGrader = Depends(grader_dependency)) -> dict:
    """Grade a cohort of embryos and rank them for review, best grade first."""
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Send at most {MAX_BATCH} images per request.")
    results = []
    for upload in files:
        try:
            results.append({"filename": upload.filename, **grader.grade_bytes(await _read(upload))})
        except InvalidImageError as exc:
            results.append({"filename": upload.filename, "error": str(exc)})

    def rank_key(item: dict):
        if "error" in item:
            return (1, 0, "Z", "Z")
        icm = (item["inner_cell_mass"] or {}).get("grade", "Z")
        te = (item["trophectoderm"] or {}).get("grade", "Z")
        return (0, -int(item["good_quality"]), icm, te)

    return {"count": len(results), "results": sorted(results, key=rank_key)}
