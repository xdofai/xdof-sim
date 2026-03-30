#!/usr/bin/env bash
# Sync latest code from upstream (abc-dataset/xdof-sim) into this repo (xdofai/xdof-sim).
# Pulls the latest upstream main, replaces local files, and commits as current git user.
#
# Usage:
#   ./sync_upstream.sh

set -euo pipefail

UPSTREAM_URL="https://github.com/abc-dataset/xdof-sim.git"
BRANCH="main"

# Ensure upstream remote exists
if ! git remote get-url upstream &>/dev/null; then
    echo "Adding upstream remote: $UPSTREAM_URL"
    git remote add upstream "$UPSTREAM_URL"
fi

# Make sure we're on main
git checkout "$BRANCH"

echo "Fetching upstream..."
git fetch upstream "$BRANCH"

# Check if there are new changes
LOCAL_HEAD=$(git rev-parse HEAD)
UPSTREAM_HEAD=$(git rev-parse "upstream/$BRANCH")

# Get the tree of upstream to compare content (ignoring our sync script)
# We use a temp dir to do the comparison
TMPDIR=$(mktemp -d)
git archive "upstream/$BRANCH" | tar -x -C "$TMPDIR"

# Preserve our sync script
cp sync_upstream.sh "$TMPDIR/sync_upstream.sh" 2>/dev/null || true

# Remove all tracked files except .git
git ls-files -z | xargs -0 rm -f

# Copy upstream content
cp -R "$TMPDIR"/. .
rm -rf "$TMPDIR"

# Stage all changes
git add -A

# Check if there are actual changes
if git diff --cached --quiet; then
    echo "Already up to date. No changes to sync."
    exit 0
fi

# Commit as current git user
GIT_USER=$(git config user.name)
GIT_EMAIL=$(git config user.email)
UPSTREAM_SHA=$(git rev-parse --short "upstream/$BRANCH")

echo "Committing as $GIT_USER <$GIT_EMAIL>..."
git commit -m "Sync upstream abc-dataset/xdof-sim@${UPSTREAM_SHA}"

echo "Pushing to origin..."
git push origin "$BRANCH"

echo "Sync complete."
