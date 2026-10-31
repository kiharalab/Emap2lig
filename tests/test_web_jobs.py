"""HTTP tests for structured job progress, the GPU queue, and cancel."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import pytest

from emap2lig.progress import emit

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from emap2lig.web.app import create_app
from emap2lig.web.state import clear_jobs


def _fake_detect(input_map, output_dir, cfg, emdb_id=None, on_progress=None):
    out = Path(output_dir)
    (out / "preprocess").mkdir(parents=True, exist_ok=True)
    (out / "preprocess" / "unified.mrc").write_bytes(b"mrc")
    blobs = out / "find_blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / "blob_1.npz").write_bytes(b"")
    emit(on_progress, "preprocess", 0, 1, "Loading map")
    emit(on_progress, "preprocess", 1, 1, "Map ready")
    emit(on_progress, "segment", 1, 1, "Segmented")
    emit(on_progress, "split_blobs", 1, 1, "Saved blob 1")
    return blobs


def _wait_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    last: dict | None = None
    while time.time() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        last = response.json()
        if last["status"] in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for job {job_id}: {last}")


@pytest.fixture
def client(tmp_path, monkeypatch):
    jobs_root = tmp_path / "jobs_output"
    monkeypatch.setattr(
        "emap2lig.web.routers.detect.validate_gpu_selection", lambda ids: None
    )
    monkeypatch.setattr(
        "emap2lig.web.routers.model.validate_gpu_selection", lambda ids: None
    )
    monkeypatch.setattr(
        "emap2lig.web.services.validate_gpu_selection", lambda ids: None
    )
    monkeypatch.setattr("emap2lig.web.services.JOBS_ROOT", jobs_root)
    monkeypatch.setattr("emap2lig.web.routers.detect.JOBS_ROOT", jobs_root)
    monkeypatch.setattr(
        "emap2lig.pipeline.detect_ligand_objects",
        _fake_detect,
    )
    clear_jobs()
    with TestClient(create_app()) as test_client:
        yield test_client
    clear_jobs()


def _start_detect(client: TestClient, contour: float = 0.1) -> str:
    response = client.post(
        "/api/detect",
        data={"gpu": "0", "contour_level": str(contour)},
        files={"input_map": ("map.mrc", b"fake-map", "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def test_detect_job_reports_stages(client: TestClient) -> None:
    job_id = _start_detect(client)
    job = _wait_job(client, job_id)
    assert job["status"] == "completed"
    assert "progress_detail" not in job
    stages = {s["stage"]: s for s in job["stages"]}
    assert set(stages) >= {"preprocess", "segment", "split_blobs"}
    assert stages["split_blobs"]["current"] == 1
    assert stages["split_blobs"]["total"] == 1
    assert job["num_blobs"] == 1
    listed = client.get("/api/jobs")
    assert listed.status_code == 200
    ids = [j["id"] for j in listed.json()]
    assert job_id in ids


def test_queue_position_and_fifo(client: TestClient, monkeypatch) -> None:
    started = Event()
    release = Event()

    def blocking_detect(input_map, output_dir, cfg, emdb_id=None, on_progress=None):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("release not signalled")
        return _fake_detect(input_map, output_dir, cfg, emdb_id, on_progress)

    monkeypatch.setattr("emap2lig.pipeline.detect_ligand_objects", blocking_detect)

    first = _start_detect(client, contour=0.1)
    assert started.wait(timeout=5)
    second = _start_detect(client, contour=0.2)
    queued = client.get(f"/api/jobs/{second}").json()
    assert queued["status"] == "queued"
    assert queued["queue_position"] == 1

    listed = {j["id"]: j for j in client.get("/api/jobs").json()}
    assert listed[second]["queue_position"] == 1
    assert listed[first]["status"] == "running"

    release.set()
    assert _wait_job(client, first)["status"] == "completed"
    assert _wait_job(client, second)["status"] == "completed"


def test_cancel_queued_and_running(client: TestClient, monkeypatch) -> None:
    started = Event()
    release = Event()

    def looping_detect(input_map, output_dir, cfg, emdb_id=None, on_progress=None):
        started.set()
        for i in range(1, 500):
            emit(on_progress, "segment", i, 500, f"window {i}")
            if release.wait(timeout=0.02):
                break
        return _fake_detect(input_map, output_dir, cfg, emdb_id, on_progress)

    monkeypatch.setattr("emap2lig.pipeline.detect_ligand_objects", looping_detect)

    running_id = _start_detect(client, contour=0.1)
    assert started.wait(timeout=5)
    queued_id = _start_detect(client, contour=0.2)

    cancelled_queued = client.post(f"/api/jobs/{queued_id}/cancel")
    assert cancelled_queued.status_code == 200
    assert cancelled_queued.json()["status"] == "cancelled"

    cancelled_running = client.post(f"/api/jobs/{running_id}/cancel")
    assert cancelled_running.status_code == 200
    job = _wait_job(client, running_id)
    assert job["status"] == "cancelled"
    release.set()


def test_missing_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    assert client.post("/api/jobs/does-not-exist/cancel").status_code == 404


def test_cancel_running_sets_cancel_requested(client: TestClient, monkeypatch) -> None:
    started = Event()
    release = Event()

    def looping_detect(input_map, output_dir, cfg, emdb_id=None, on_progress=None):
        started.set()
        for i in range(1, 500):
            emit(on_progress, "segment", i, 500, f"window {i}")
            if release.wait(timeout=0.02):
                break
        return _fake_detect(input_map, output_dir, cfg, emdb_id, on_progress)

    monkeypatch.setattr("emap2lig.pipeline.detect_ligand_objects", looping_detect)

    job_id = _start_detect(client)
    assert started.wait(timeout=5)
    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["cancel_requested"] is True
    job = _wait_job(client, job_id)
    assert job["status"] == "cancelled"
    running = [s for s in job["stages"] if s["status"] == "running"]
    assert running == []
    release.set()


def test_cancel_finished_job_is_400(client: TestClient) -> None:
    job_id = _start_detect(client)
    assert _wait_job(client, job_id)["status"] == "completed"
    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 400


def test_cancel_running_model_job(client: TestClient, monkeypatch) -> None:
    started = Event()
    release = Event()

    def looping_model(job, ligand_list_path, blob_ids=None):
        from emap2lig.web.services import job_progress_callback

        started.set()
        callback = job_progress_callback(job)
        for i in range(1, 500):
            emit(callback, "predict", i, 500, f"step {i}")
            if release.wait(timeout=0.02):
                break
        out = Path(job.output_dir) / "build_struct"
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr("emap2lig.web.services._model_sync", looping_model)

    detect_id = _start_detect(client)
    assert _wait_job(client, detect_id)["status"] == "completed"
    response = client.post(
        "/api/model",
        json={
            "job_id": detect_id,
            "ligand_list": [{"type": "CCD", "name": "NAG"}],
            "gpu": [0],
        },
    )
    assert response.status_code == 200, response.text
    model_id = response.json()["job_id"]
    assert started.wait(timeout=5)
    cancelled = client.post(f"/api/jobs/{model_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["cancel_requested"] is True
    job = _wait_job(client, model_id)
    assert job["status"] == "cancelled"
    release.set()
