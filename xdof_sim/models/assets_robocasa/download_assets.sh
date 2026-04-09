#!/bin/bash
# Download RoboCasa assets required for the inhand_transfer environment.
# Only fetches the two packs needed (~220MB unzipped):
#   - objects_lightwheel  (dish_brush, whisk, salt_and_pepper_shaker, ...)
#   - objaverse           (rolling_pin, water_bottle, can, ladle)
#
# Run from this directory:
#   cd xdof_sim/models/assets_robocasa
#   bash download_assets.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

declare -A PACKS=(
    ["objects_lightwheel"]="vckqvvkh1z8t69k8qcpcmee6k66stii4"
    ["objaverse"]="03eionyo8fk3a9dsksq9jb8du5lqfw8h"
)

for name in "${!PACKS[@]}"; do
    id="${PACKS[$name]}"
    zip_file="${name}.zip"
    url="https://utexas.box.com/shared/static/${id}.zip"

    if [ -d "$name" ]; then
        echo "--- Skipping ${name} (already exists) ---"
        continue
    fi

    echo "--- Downloading ${name} ---"
    wget -O "$zip_file" "$url"

    echo "--- Unzipping ${name} ---"
    mkdir -p "$name"
    unzip "$zip_file" -d "$name"
    rm "$zip_file"
done

echo "Done."
