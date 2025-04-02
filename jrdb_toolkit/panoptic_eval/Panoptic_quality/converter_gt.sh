
#!/bin/bash
# Define the directory containing the JSON files
INPUT_DIR="panoptic_stitched folder"
OUTPUT_DIR="coco panoptic format stitched folder "
CATEGORIES_JSON="jrdb_categories_cocoformat.json"
# Loop over each JSON file in the input directory
for input_file in "$INPUT_DIR"/*.json; do
    # Extract the base filename without path and extension
    filename=$(basename -- "$input_file")
    base="${filename%%.*}"

    # Define output JSON file path
    output_json_file="$OUTPUT_DIR/${base}.json"

    # Run the Python script with the necessary arguments
    python detection2panoptic_coco_format.py \
      --input_json_file "$input_file" \
      --categories_json_file "$CATEGORIES_JSON" \
      --output_json_file "$output_json_file" \
      --non_overlap
done
