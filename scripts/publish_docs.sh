#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE_DIR="${ROOT_DIR}/docs/_site"
WORKTREE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pyensmallen-gh-pages.XXXXXX")"

cleanup() {
  if git -C "${ROOT_DIR}" worktree list | grep -q "${WORKTREE_DIR}"; then
    git -C "${ROOT_DIR}" worktree remove --force "${WORKTREE_DIR}"
  fi
  rm -rf "${WORKTREE_DIR}"
}

trap cleanup EXIT

if [ ! -d "${SITE_DIR}" ]; then
  echo "docs/_site does not exist. Render the documentation site before publishing." >&2
  exit 1
fi

git -C "${ROOT_DIR}" worktree add "${WORKTREE_DIR}" gh-pages
rsync -a --delete --exclude .git "${SITE_DIR}/" "${WORKTREE_DIR}/"
touch "${WORKTREE_DIR}/.nojekyll"

git -C "${WORKTREE_DIR}" add -A

if git -C "${WORKTREE_DIR}" diff --cached --quiet; then
  echo "No site changes to publish."
  exit 0
fi

git -C "${WORKTREE_DIR}" commit -m "Update published docs site"
git -C "${WORKTREE_DIR}" push origin gh-pages

echo "Published docs/_site to gh-pages."
