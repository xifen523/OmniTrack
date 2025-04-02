#!/bin/bash

# Define the paths for the directories and files
#DIRECTORY_PATH="Contains all individual view prediction JSONs, e.g., cubberly-auditorium-2019-04-22_1_camera_0, cubberly-auditorium-2019-04-22_1_camera_1, etc."
#OUTPUT_DIR="Output folder: will contain files such as cubberly-auditorium-2019-04-22_1.json, discovery-walk-2019-02-28_0.json (each file is stitched annotations from 5 individual views)"
#CALIBRATION_DIR="Parent directory contains the 'indi2stitch_mappings' folder, which includes files such as indi2stitch_mapping_camera_0.npy, indi2stitch_mapping_camera_2.npy, etc."
CATEGORIES_JSON_FILE="" # The JSON file contains all JRDB categories
DIRECTORY_PATH=""
OUTPUT_DIR=""
CALIBRATION_DIR=""
# Execute the Python script
python generate_stitched_prediction_v2.py \
--directory_path "$DIRECTORY_PATH" \
--output_dir "$OUTPUT_DIR" \
--calibration_dir "$CALIBRATION_DIR" \
--categories_json_file "$CATEGORIES_JSON_FILE"