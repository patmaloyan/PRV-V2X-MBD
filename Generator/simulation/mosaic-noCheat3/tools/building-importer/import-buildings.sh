#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOSAIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCENARIO_DIR="${1:-${MOSAIC_ROOT}/scenarios/urban}"
BUILD_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

CLASSPATH="$(find "${MOSAIC_ROOT}/lib" -type f -name '*.jar' -print | paste -sd: -)"

javac \
    -cp "${CLASSPATH}" \
    -d "${BUILD_DIR}" \
    "${SCRIPT_DIR}/src/ImportBuildings.java"

java \
    -cp "${BUILD_DIR}:${CLASSPATH}" \
    ImportBuildings \
    "${SCENARIO_DIR}"
