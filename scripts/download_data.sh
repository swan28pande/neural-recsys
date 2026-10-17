#!/usr/bin/env bash
# Download and extract MovieLens-1M dataset.
#
# Usage:
#   bash scripts/download_data.sh [output_dir]
#
# The dataset is provided by GroupLens Research:
#   https://grouplens.org/datasets/movielens/1m/

set -euo pipefail

DATA_DIR="${1:-data}"
URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP_FILE="${DATA_DIR}/ml-1m.zip"

echo "=== MovieLens-1M Dataset Download ==="
echo "Output directory: ${DATA_DIR}"

mkdir -p "${DATA_DIR}"

if [ -d "${DATA_DIR}/ml-1m" ] && [ -f "${DATA_DIR}/ml-1m/ratings.dat" ]; then
    echo "Dataset already exists at ${DATA_DIR}/ml-1m. Skipping download."
    exit 0
fi

echo "Downloading MovieLens-1M from ${URL}..."
curl -L -o "${ZIP_FILE}" "${URL}"

echo "Extracting..."
unzip -o "${ZIP_FILE}" -d "${DATA_DIR}"

echo "Cleaning up..."
rm -f "${ZIP_FILE}"

echo "Done! Dataset files:"
ls -la "${DATA_DIR}/ml-1m/"
echo ""
echo "Files:"
echo "  ratings.dat  - UserID::MovieID::Rating::Timestamp"
echo "  users.dat    - UserID::Gender::Age::Occupation::Zip-code"
echo "  movies.dat   - MovieID::Title::Genres"
echo ""
echo "Statistics:"
wc -l "${DATA_DIR}/ml-1m/ratings.dat" | awk '{print "  Ratings: " $1}'
wc -l "${DATA_DIR}/ml-1m/users.dat"   | awk '{print "  Users:   " $1}'
wc -l "${DATA_DIR}/ml-1m/movies.dat"  | awk '{print "  Movies:  " $1}'
