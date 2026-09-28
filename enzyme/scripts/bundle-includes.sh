#!/bin/bash

set -e

INPUT_DIR=$(dirname "$1")/..
OUTPUT_FILE=$2
MAX_LINE_CHUNK=8000

echo "$INPUT_DIR"
echo "$OUTPUT_FILE"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo > "$OUTPUT_FILE"
echo "const char* include_headers[][2] = {" >> "$OUTPUT_FILE"
find -L "$INPUT_DIR" -type f -print0 | while IFS= read -r -d '' FILE; do
    echo "$FILE"
    INTERNAL_FILENAME="/enzymeroot/${FILE#"$INPUT_DIR"/}"
    echo "$INTERNAL_FILENAME"
    echo '{"'"$INTERNAL_FILENAME"'",' >> "$OUTPUT_FILE"
    printf 'R"(' >> "$OUTPUT_FILE"
    LINE_COUNT=0
    while IFS= read -r LINE || [ -n "$LINE" ]; do
        while [ ${#LINE} -gt $MAX_LINE_CHUNK ]; do
            printf '%s' "${LINE:0:$MAX_LINE_CHUNK}" >> "$OUTPUT_FILE"
            printf ')"\nR"(' >> "$OUTPUT_FILE"
            LINE="${LINE:$MAX_LINE_CHUNK}"
        done
        printf '%s\n' "$LINE" >> "$OUTPUT_FILE"
        LINE_COUNT=$((LINE_COUNT + 1))
        if [ "$LINE_COUNT" -ge 200 ]; then
            echo ')"' >> "$OUTPUT_FILE"
            printf 'R"(' >> "$OUTPUT_FILE"
            LINE_COUNT=0
        fi
    done < "$FILE"
    echo ')"' >> "$OUTPUT_FILE"
    echo '},' >> "$OUTPUT_FILE"
done

echo '};' >>"${OUTPUT_FILE}"
