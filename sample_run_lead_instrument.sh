#!/bin/bash
# Sample run: Generate an instrumental track with a lead instrument
# Usage: bash sample_run_lead_instrument.sh

# Example 1: Rock instrumental with electric guitar lead
python3 main.py \
  --genre rock \
  --mood energetic \
  --tempo 140 \
  --duration 3 \
  --instruments "electric guitar,drums,bass" \
  --vocals false

# Example 2: Jazz instrumental with piano lead
# Uncomment to try jazz piano lead
# python3 music_generator/main.py \
#   --genre jazz \
#   --mood smooth \
#   --tempo 120 \
#   --duration 2 \
#   --instruments "piano,bass,drums" \
#   --vocals false 