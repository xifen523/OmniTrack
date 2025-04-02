#!/bin/bash

### INPUT_DIR contains your prediction json files
### OUTPUT_DIR contains your prediction result
### CATEGORIES_JSON contains all jrdb_categories
### GT_DIR contains all groundtruth json and groundtruth pngs

INPUT_DIR=""
OUTPUT_DIR=""
CATEGORIES_JSON="jrdb_categories_cocoformat.json"
GT_DIR=""
SAVE_PREDICTION_DIR="$OUTPUT_DIR/pq_calculation"
echo "saving to this$SAVE_PREDICTION_DIR"
# Ensure the output directory exists
mkdir -p "$SAVE_PREDICTION_DIR"
# Loop over each JSON file in the input directory
for input_file in "$INPUT_DIR"/*.json; do
    # Extract the filename and base name from the input file path
    filename=$(basename -- "$input_file")
    base="${filename%%.*}"

    # Define output JSON file path using the same base name as the input file
    output_json_file="$OUTPUT_DIR/${filename}"

    # Define ground truth JSON file path using the same base name
    gt_json_file="$GT_DIR/${filename}"

    # Convert detection to COCO format panoptic JSON
    python detection2panoptic_coco_format.py \
        --input_json_file "$input_file" \
        --categories_json_file "$CATEGORIES_JSON" \
        --output_json_file "$output_json_file"

    # Run evaluation script and save the results
    python evaluation.py \
        --gt_json_file "$gt_json_file" \
        --pred_json_file "$output_json_file" \
        --pq_saving_path "$SAVE_PREDICTION_DIR" \
        --OW ###If open world
done

# Wait for all background processes to finish
wait

# Run final processing script
python generate_all_result.py \
    --pred_dir "$SAVE_PREDICTION_DIR" \
    --CSV \
    --OW ###If open world
# Delete all files in OUTPUT_DIR except those in the "$OUTPUT_DIR/pq" folder
find "$OUTPUT_DIR" -mindepth 1 -not -path "$OUTPUT_DIR/pq" -not -path "$OUTPUT_DIR/pq/*" -delete

echo "Cleanup complete, kept files only in $OUTPUT_DIR/pq"
    #--OW