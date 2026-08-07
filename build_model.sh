#!/usr/bin/env bash
# Build model/bge-reranker-v2-m3/ — the local model directory TEI serves
# (--model-id). Same job as ../tei/build_model.sh, and the same two reasons
# (see README.md):
#  - caps max_seq_length, which is what bounds TEI's memory on CPU
#  - no outbound weight download at startup (ARCHITECTURE.md block 5)
#
# It downloads straight from the hub rather than scraping a cache a hub-backed
# TEI run left behind, which is how ../tei/build_model.sh works. That is
# deliberate: the cache-populating run serves the model UNCAPPED, and this
# backbone uncapped is the ~12 GB warmup that block 5 documents — the exact
# thing the cap exists to prevent. Bootstrapping the fix by first triggering the
# bug is a bad trade when six files over HTTPS do the same job. It also pins the
# revision, so a rebuild is reproducible.
#
# Two more differences from the embedder's copy, both because the upstream repo
# is shaped differently:
#  - no ONNX export is published, so the safetensors weights are what gets
#    placed (TEI's CPU backend loads them directly)
#  - no sentence_bert_config.json ships at all, so this WRITES one rather than
#    overwriting it. Without it TEI falls back to config.json's
#    max_position_embeddings (8194) and the cap never applies — check the
#    startup log, see README.md.
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-BAAI/bge-reranker-v2-m3}"
# Pinned so a rebuild is reproducible; bump deliberately, then re-measure memory.
REVISION="${REVISION:-953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1536}"
DEST="model/bge-reranker-v2-m3"

FILES=(
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    sentencepiece.bpe.model
    model.safetensors
)

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }

TMP="$(mktemp -d "${DEST%/*}/.build.XXXXXX" 2>/dev/null || mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading $MODEL @ ${REVISION:0:12} (~2.3 GB)"
for f in "${FILES[@]}"; do
    printf '  %-26s' "$f"
    curl -fsSL --retry 3 --retry-delay 2 \
        "https://huggingface.co/$MODEL/resolve/$REVISION/$f" -o "$TMP/$f"
    printf '%s\n' "$(du -h "$TMP/$f" | cut -f1)"
done

# The one addition. Upstream publishes no sentence_bert_config.json, so TEI would
# read config.json's max_position_embeddings (8194) instead. Reranking scores
# query+passage together, so the budget has to cover both: the chunker's longest
# measured output is 1257 tokens, leaving ~280 for the query at 1536.
printf '{\n  "max_seq_length": %s,\n  "do_lower_case": false\n}\n' "$MAX_SEQ_LENGTH" \
    > "$TMP/sentence_bert_config.json"

cat > "$TMP/PROVENANCE" <<EOF
model:          $MODEL
revision:       $REVISION
max_seq_length: $MAX_SEQ_LENGTH (upstream: 8192, via max_position_embeddings 8194)
note:           sentence_bert_config.json is written by build_model.sh, not upstream
built:          $(date -u +%Y-%m-%dT%H:%M:%SZ) by build_model.sh
EOF

# Swap in whole, so an interrupted download never leaves a half-model that TEI
# would try to serve.
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$TMP" "$DEST"
trap - EXIT
chmod -R a+rX "$DEST"

echo "built $DEST from ${REVISION:0:12} (max_seq_length=$MAX_SEQ_LENGTH)"
du -sh "$DEST"
