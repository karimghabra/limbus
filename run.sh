#!/bin/bash
# Launch the Basler Video Recorder
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/venv/bin/python" "$DIR/camera_recorder.py" "$@"
