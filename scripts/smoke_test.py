#!/usr/bin/env python3

"""scripts/smoke_test.py
Simple end‑to‑end smoke test for the FastAPI backend.
It starts the backend using the helper script, waits for it to be ready,
then verifies a few key endpoints.
"""

import subprocess
import time
import sys
import os
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PY = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
RUN_BACKEND = os.path.join(PROJECT_ROOT, "scripts", "run_backend.sh")

def start_backend():
    # Run the backend in a subprocess (non‑blocking)
    proc = subprocess.Popen(["bash", RUN_BACKEND], cwd=PROJECT_ROOT)
    return proc

def wait_for_health(url, timeout=30):
    for _ in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    backend_proc = start_backend()
    try:
        health_url = "http://127.0.0.1:8000/health"
        if not wait_for_health(health_url):
            print("[ERROR] Backend health check failed.")
            sys.exit(1)
        print("[INFO] Backend is healthy.")

        # Create a session
        sess_resp = requests.post("http://127.0.0.1:8000/session")
        sess_resp.raise_for_status()
        session_id = sess_resp.json()["session_id"]
        print(f"[INFO] Created session {session_id}")

        # Send a dummy message
        msg_payload = {"content": "Test message for smoke test"}
        msg_resp = requests.post(
            f"http://127.0.0.1:8000/session/{session_id}/message",
            json=msg_payload,
        )
        msg_resp.raise_for_status()
        print("[INFO] Message endpoint responded:", msg_resp.json())

        # Try fetching requirements (may be empty)
        req_resp = requests.get(f"http://127.0.0.1:8000/requirements/{session_id}")
        if req_resp.status_code == 200:
            print("[INFO] Requirements endpoint ok:", req_resp.json())
        else:
            print("[WARN] Requirements endpoint returned", req_resp.status_code)

        # Try fetching SDLC recommendation (may be empty)
        sdlc_resp = requests.get(f"http://127.0.0.1:8000/sdlc/{session_id}")
        if sdlc_resp.status_code == 200:
            print("[INFO] SDLC endpoint ok:", sdlc_resp.json())
        else:
            print("[WARN] SDLC endpoint returned", sdlc_resp.status_code)

        print("[SUCCESS] Smoke test completed.")
    finally:
        # Terminate backend gracefully
        backend_proc.terminate()
        backend_proc.wait()
        print("[INFO] Backend process stopped.")

if __name__ == "__main__":
    main()
