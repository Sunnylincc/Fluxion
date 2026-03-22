#!/usr/bin/env bash
uvicorn fluxion.api.server:app --host 0.0.0.0 --port 8000
