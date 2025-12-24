#!/bin/bash
PORT=${PORT:-8000}  # use 8000 if $PORT is not set
uvicorn backend:app --host 0.0.0.0 --port $PORT
