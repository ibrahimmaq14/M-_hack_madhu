"""FastAPI service for laptop 1."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import load_reference, trip_ids
from .evaluation import route_metrics
from .features import align_reference
from .service import Matcher, public_prediction

REPO = Path(__file__).resolve().parents[1]
logger = logging.getLogger("uvicorn.error")


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{8}$")


def create_app():
    trips_root = Path(os.environ.get(
        "PATHSNAP_TRIPS", str(REPO.parent / "Student folder" / "Student folder" / "student_test")
    ))
    references_root = Path(os.environ.get("PATHSNAP_REFERENCES", str(trips_root)))
    artifact_dir = Path(os.environ.get("PATHSNAP_ARTIFACTS", str(REPO / "artifacts")))
    run_mode = os.environ.get("PATHSNAP_MODE", "student_demo")
    origins = os.environ.get(
        "PATHSNAP_UI_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080"
    ).split(",")

    @asynccontextmanager
    async def lifespan(app):
        app.state.matcher = Matcher(artifact_dir)
        app.state.trip_ids = trip_ids(trips_root)
        if not app.state.trip_ids:
            raise RuntimeError("No trip folders contain .track, .nodes and .arcs")
        logger.info("PathSnap ready: mode=%s trips=%d", run_mode, len(app.state.trip_ids))
        yield

    api = FastAPI(title="PathSnap", version="1.0.0", lifespan=lifespan)
    api.add_middleware(
        CORSMiddleware, allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    @api.get("/health")
    def health():
        return {"status": "ready", "mode": run_mode, "model": api.state.matcher.metadata["model"],
                "variant": api.state.matcher.metadata["variant"], "trip_count": len(api.state.trip_ids)}

    @api.get("/trips")
    def trips():
        return {"mode": run_mode, "trips": api.state.trip_ids}

    @api.post("/predict")
    def predict(request: PredictRequest):
        trip_id = request.trip_id
        if trip_id not in api.state.trip_ids:
            raise HTTPException(status_code=404, detail="Trip is unavailable in this partition.")
        started = perf_counter()
        try:
            trip, candidates, matched = api.state.matcher.match(trips_root, trip_id)
            response = public_prediction(trip, candidates, matched)
        except (OSError, ValueError):
            logger.exception("Inference failed for trip %s", trip_id)
            raise HTTPException(status_code=422, detail="The server could not process this trip.")
        inference_s = perf_counter() - started
        try:
            reference = load_reference(references_root, trip)
            aligned = align_reference(trip, reference)
            metrics = route_metrics(trip.graph, aligned, reference, matched.arcs, matched.route_parts)
            # Ship only aggregate scores. The ground-truth route and derived labels stay here.
            response["metrics"] = {
                name: metrics[name] for name in ("point_accuracy", "sequence_similarity", "route_continuity")
            }
        except FileNotFoundError:
            response["metrics"] = None
        except (ValueError, OSError):
            logger.exception("Reference scoring failed for trip %s", trip_id)
            response["metrics"] = None
        response["inference_seconds"] = round(inference_s, 4)
        response["elapsed_seconds"] = round(perf_counter() - started, 4)
        scores = response["metrics"] or {}
        accuracy = f"{scores['point_accuracy']:.4f}" if scores else "unscored"
        logger.info("POST /predict trip=%s status=200 points=%d accuracy=%s elapsed=%.3fs",
                    trip_id, response["observations"], accuracy, response["elapsed_seconds"])
        return response

    return api


app = create_app()
