#!/bin/bash
# Camera Tutor cleanup — releases camera and audio devices
# Run if "python3 demo.py" was killed abnormally

echo "Cleaning up Camera Tutor resources..."

# Find and kill any remaining demo processes
pkill -f "demo.py" 2>/dev/null && echo "  Killed demo.py" || echo "  No demo.py running"

# Release camera (find process using /dev/video0 and kill if stale)
for dev in /dev/video0 /dev/video1; do
    PID=$(sudo fuser $dev 2>/dev/null | cut -d: -f2)
    if [ -n "$PID" ]; then
        sudo kill -9 $PID 2>/dev/null && echo "  Released $dev from PID $PID"
    fi
done

echo "Done. Safe to re-run demo.py"
