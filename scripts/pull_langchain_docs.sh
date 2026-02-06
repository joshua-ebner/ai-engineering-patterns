#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_DIR="data/raw/langchain"
TMP_DIR="tmp/langchain-docs"

rm -rf "$TMP_DIR"
mkdir -p "$TARGET_DIR"
mkdir -p tmp

echo "Cloning LangChain docs..."
git clone --depth 1 https://github.com/langchain-ai/docs.git "$TMP_DIR"

echo "Copying docs..."
cp -R "$TMP_DIR/src/oss/langchain/." "$TARGET_DIR/"

# Light cleanup
rm -rf "$TARGET_DIR/integrations" "$TARGET_DIR/deprecated" 2>/dev/null || true

rm -rf "$TMP_DIR"

echo "Done. Docs in $TARGET_DIR"
