#!/bin/bash

# Usage: ./run_track.sh Montreal
# This script runs: python3 main.py ./racetracks/Montreal.csv ./racetracks/Montreal_raceline.csv

TRACK_NAME="$1"

if [ -z "$TRACK_NAME" ]; then
    echo "Error: No track name provided."
    echo "Usage: ./run_track.sh <TrackName>"
    exit 1
fi

TRACK_FILE="./racetracks/${TRACK_NAME}.csv"
RACELINE_FILE="./racetracks/${TRACK_NAME}_raceline.csv"

if [ ! -f "$TRACK_FILE" ]; then
    echo "Error: Track file not found: $TRACK_FILE"
    exit 1
fi

if [ ! -f "$RACELINE_FILE" ]; then
    echo "Error: Raceline file not found: $RACELINE_FILE"
    exit 1
fi

echo "Running track: $TRACK_NAME"
python3 main.py "$TRACK_FILE" "$RACELINE_FILE"
