"""Tests for GET /api/task_history endpoint."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Patch NOTE_OUTPUT_DIR before importing the router
_tmpdir = tempfile.mkdtemp()


@pytest.fixture(autouse=True)
def patch_output_dir(tmp_path):
    """Redirect NOTE_OUTPUT_DIR to a temp dir for each test."""
    with patch("app.routers.note.NOTE_OUTPUT_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def client():
    from app import create_app
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    app = create_app(noop_lifespan)
    return TestClient(app)


def _write_json(directory, filename, content):
    path = os.path.join(str(directory), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f)
    return path


class TestTaskHistory:
    def test_empty_directory(self, client, patch_output_dir):
        """Empty note_results/ returns empty list."""
        resp = client.get("/api/task_history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert body["data"]["tasks"] == []
        assert body["data"]["total"] == 0

    def test_returns_result_files(self, client, patch_output_dir):
        """Returns only main result JSON files."""
        result = {"markdown": "# Hello", "transcript": {}, "audio_meta": {}}
        _write_json(patch_output_dir, "aaa-111.json", result)
        # Auxiliary files should be excluded
        _write_json(patch_output_dir, "aaa-111_audio.json", {"path": "/tmp/a.mp3"})
        _write_json(patch_output_dir, "aaa-111_transcript.json", {"segments": []})
        _write_json(patch_output_dir, "aaa-111.status.json", {"status": "SUCCESS"})

        resp = client.get("/api/task_history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["total"] == 1
        assert len(body["data"]["tasks"]) == 1
        task = body["data"]["tasks"][0]
        assert task["task_id"] == "aaa-111"
        assert task["status"] == "SUCCESS"
        assert task["result"]["markdown"] == "# Hello"

    def test_pagination(self, client, patch_output_dir):
        """Pagination with limit and offset works correctly."""
        for i in range(5):
            result = {"markdown": f"Note {i}"}
            path = _write_json(patch_output_dir, f"task-{i:03d}.json", result)
            # Set different mtime so ordering is deterministic
            os.utime(path, (1000000 + i, 1000000 + i))

        # Default returns all 5
        resp = client.get("/api/task_history")
        assert resp.json()["data"]["total"] == 5

        # limit=2, offset=0 → first 2 (newest first, so task-004, task-003)
        resp = client.get("/api/task_history?limit=2&offset=0")
        body = resp.json()["data"]
        assert len(body["tasks"]) == 2
        assert body["tasks"][0]["task_id"] == "task-004"
        assert body["tasks"][1]["task_id"] == "task-003"
        assert body["total"] == 5
        assert body["limit"] == 2
        assert body["offset"] == 0

        # offset=3 → last 2 (task-001, task-000)
        resp = client.get("/api/task_history?limit=2&offset=3")
        body = resp.json()["data"]
        assert len(body["tasks"]) == 2
        assert body["tasks"][0]["task_id"] == "task-001"
        assert body["tasks"][1]["task_id"] == "task-000"

    def test_corrupted_json_skipped(self, client, patch_output_dir):
        """Corrupted JSON files are skipped without error."""
        # Valid file
        _write_json(patch_output_dir, "good-task.json", {"markdown": "ok"})
        # Corrupted file
        bad_path = os.path.join(str(patch_output_dir), "bad-task.json")
        with open(bad_path, "w") as f:
            f.write("{invalid json content!!")

        resp = client.get("/api/task_history")
        assert resp.status_code == 200
        body = resp.json()
        # total counts files on disk (before parsing), but tasks only has valid ones
        assert body["data"]["total"] == 2
        assert len(body["data"]["tasks"]) == 1
        assert body["data"]["tasks"][0]["task_id"] == "good-task"

    def test_nonexistent_directory(self, client):
        """If note_results/ doesn't exist, returns empty list."""
        with patch("app.routers.note.NOTE_OUTPUT_DIR", "/nonexistent/path/xyz"):
            resp = client.get("/api/task_history")
            assert resp.status_code == 200
            body = resp.json()
            assert body["data"]["tasks"] == []
            assert body["data"]["total"] == 0
