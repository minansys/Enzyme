#!/bin/bash

set -e

INPUT_DIR=$(dirname "${1}")/..
OUTPUT_FILE=${2}
MAX_LINE_CHUNK=8000

echo "${INPUT_DIR}"
echo "${OUTPUT_FILE}"
mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo >"${OUTPUT_FILE}"
echo "const char* include_headers[][2] = {" >>"${OUTPUT_FILE}"

find -L "${INPUT_DIR}" -type f | while IFS= read -r FILE; do
  echo "${FILE}"
  INTERNAL_FILENAME="${FILE/${INPUT_DIR}/\/enzymeroot\/}"
  echo "${INTERNAL_FILENAME}"
  {
    echo '{"'"${INTERNAL_FILENAME}"'",'
    echo 'R"('
    LINE_COUNT=0
    while IFS= read -r LINE || [ -n "${LINE}" ]; do
      while [ ${#LINE} -gt "$MAX_LINE_CHUNK" ]; do
        printf '%s' "${LINE:0:$MAX_LINE_CHUNK}"
        echo ')"'
        printf '%s' 'R"('
        LINE="${LINE:$MAX_LINE_CHUNK}"
      done
      printf '%s\n' "${LINE}"
      LINE_COUNT=$((LINE_COUNT + 1))
      if [ "$LINE_COUNT" -ge 200 ]; then
        echo ')"'
        printf '%s' 'R"('
        LINE_COUNT=0
      fi
    done < "${FILE}"
    echo ')"'
    echo '},'
  } >>"${OUTPUT_FILE}"
done

echo '};' >>"${OUTPUT_FILE}"
