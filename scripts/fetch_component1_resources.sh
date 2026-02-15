#!/usr/bin/env bash
#
# fetch_component1_resources.sh
# 
# Fetches external resources (papers + repos) for Component 1 implementation.
# Idempotent: safe to re-run (won't re-download if already present).
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPERS_DIR="${REPO_ROOT}/resources/papers"
CODE_DIR="${REPO_ROOT}/resources/external_code"
MANIFEST_DIR="${REPO_ROOT}/resources/manifests"

echo "=== Component 1 Resource Fetcher ==="
echo "Repo root: ${REPO_ROOT}"
echo ""

# Ensure directories exist
mkdir -p "${PAPERS_DIR}" "${CODE_DIR}" "${MANIFEST_DIR}"

# Array to track downloaded resources
declare -A MANIFEST_DATA

# -----------------------------------------------------------------------------
# PAPERS
# -----------------------------------------------------------------------------
echo "[1/2] Fetching papers..."

# Helper function to download PDFs
download_pdf() {
    local name="$1"
    local url="$2"
    local output="${PAPERS_DIR}/${name}.pdf"
    
    if [[ -f "${output}" ]]; then
        echo "  ✓ Already exists: ${name}.pdf"
        MANIFEST_DATA["paper_${name}"]="${output}|${url}|cached"
    else
        echo "  ⬇ Downloading: ${name}.pdf"
        if curl -L -o "${output}" "${url}" --fail --silent --show-error; then
            echo "    ✓ Downloaded successfully"
            MANIFEST_DATA["paper_${name}"]="${output}|${url}|downloaded"
        else
            echo "    ✗ Failed to download from ${url}"
            MANIFEST_DATA["paper_${name}"]="FAILED|${url}|error"
        fi
    fi
}

# Opsahl Fact-or-Fiction (arXiv)
download_pdf "opsahl_fact_or_fiction" "https://arxiv.org/pdf/2408.07453.pdf"

# FEVER workshop paper (ACL Anthology)
download_pdf "fever_workshop" "https://aclanthology.org/2024.fever-1.32.pdf"

# FactKG paper (arXiv)
download_pdf "factkg" "https://arxiv.org/pdf/2305.06590.pdf"

# FEVER paper (ACL Anthology N18-1074)
download_pdf "fever_original" "https://aclanthology.org/N18-1074.pdf"

# HoVer paper (ACL Anthology Findings EMNLP 2020)
download_pdf "hover" "https://aclanthology.org/2020.findings-emnlp.309.pdf"

# CO-GAT paper (arXiv)
download_pdf "cogat" "https://arxiv.org/pdf/2405.10481.pdf"

# DeBERTaV3 paper (arXiv)
download_pdf "debertav3" "https://arxiv.org/pdf/2111.09543.pdf"

echo ""

# -----------------------------------------------------------------------------
# REPOSITORIES
# -----------------------------------------------------------------------------
echo "[2/2] Cloning repositories..."

# Helper function to clone repos
clone_repo() {
    local name="$1"
    local url="$2"
    local target="${CODE_DIR}/${name}"
    
    if [[ -d "${target}/.git" ]]; then
        echo "  ✓ Already cloned: ${name}"
        pushd "${target}" > /dev/null
        local commit_hash=$(git rev-parse HEAD)
        local remote_url=$(git config --get remote.origin.url || echo "${url}")
        MANIFEST_DATA["repo_${name}"]="${target}|${remote_url}|${commit_hash}|cached"
        popd > /dev/null
    else
        echo "  ⬇ Cloning: ${name}"
        if git clone "${url}" "${target}" --quiet; then
            echo "    ✓ Cloned successfully"
            pushd "${target}" > /dev/null
            local commit_hash=$(git rev-parse HEAD)
            MANIFEST_DATA["repo_${name}"]="${target}|${url}|${commit_hash}|cloned"
            popd > /dev/null
        else
            echo "    ✗ Failed to clone ${url}"
            MANIFEST_DATA["repo_${name}"]="FAILED|${url}|N/A|error"
        fi
    fi
}

# Current repo (record commit hash only)
echo "  ℹ Recording current repo commit..."
pushd "${REPO_ROOT}" > /dev/null
CURRENT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
MANIFEST_DATA["repo_fact_or_fiction"]="${REPO_ROOT}|local|${CURRENT_COMMIT}|current"
popd > /dev/null
echo "    Current commit: ${CURRENT_COMMIT}"

# FactKG repo
clone_repo "FactKG" "https://github.com/jiho283/FactKG.git"

# FEVER repo
clone_repo "FEVER" "https://github.com/awslabs/fever.git"

# HoVer repo
clone_repo "HoVer" "https://github.com/hover-nlp/hover.git"

# CO-GAT repo
clone_repo "CO-GAT" "https://github.com/neuir/co-gat.git"

# DeBERTa repo (optional but useful)
clone_repo "DeBERTa" "https://github.com/microsoft/DeBERTa.git"

echo ""

# -----------------------------------------------------------------------------
# MANIFEST GENERATION
# -----------------------------------------------------------------------------
echo "[3/3] Generating manifest..."

MANIFEST_FILE="${MANIFEST_DIR}/component1_resources.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Build JSON manually (avoiding jq dependency)
cat > "${MANIFEST_FILE}" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "repo_root": "${REPO_ROOT}",
  "papers": {
EOF

# Papers
first=true
for key in "${!MANIFEST_DATA[@]}"; do
    if [[ $key == paper_* ]]; then
        name="${key#paper_}"
        IFS='|' read -r path url status <<< "${MANIFEST_DATA[$key]}"
        
        [[ "$first" == "true" ]] || echo "," >> "${MANIFEST_FILE}"
        cat >> "${MANIFEST_FILE}" <<EOF
    "${name}": {
      "path": "${path}",
      "url": "${url}",
      "status": "${status}"
    }
EOF
        first=false
    fi
done

cat >> "${MANIFEST_FILE}" <<EOF

  },
  "repositories": {
EOF

# Repos
first=true
for key in "${!MANIFEST_DATA[@]}"; do
    if [[ $key == repo_* ]]; then
        name="${key#repo_}"
        IFS='|' read -r path url commit status <<< "${MANIFEST_DATA[$key]}"

        
        [[ "$first" == "true" ]] || echo "," >> "${MANIFEST_FILE}"
        cat >> "${MANIFEST_FILE}" <<EOF
    "${name}": {
      "path": "${path}",
      "url": "${url}",
      "commit_hash": "${commit}",
      "status": "${status}"
    }
EOF
        first=false
    fi
done

cat >> "${MANIFEST_FILE}" <<EOF

  }
}
EOF

echo "  ✓ Manifest saved: ${MANIFEST_FILE}"
echo ""
echo "=== Resource fetch complete ==="
echo "Run: cat ${MANIFEST_FILE} | python -m json.tool"
echo ""
