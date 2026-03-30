#!/usr/bin/env bash
# Sync latest changes from upstream (abc-dataset/xdof-sim) into this repo (xdofai/xdof-sim).
#
# Usage:
#   ./sync_upstream.sh          # sync default branch
#   ./sync_upstream.sh --all    # sync all branches and tags

set -euo pipefail

UPSTREAM_URL="https://github.com/abc-dataset/xdof-sim.git"

# Ensure upstream remote exists
if ! git remote get-url upstream &>/dev/null; then
    echo "Adding upstream remote: $UPSTREAM_URL"
    git remote add upstream "$UPSTREAM_URL"
fi

echo "Fetching upstream..."
git fetch upstream

# Detect upstream default branch
UPSTREAM_DEFAULT=$(git remote show upstream | sed -n 's/.*HEAD branch: //p')
if [[ -z "$UPSTREAM_DEFAULT" ]]; then
    echo "Warning: Could not detect upstream default branch. Upstream repo may be empty."
    exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
    # Sync all upstream branches
    for ref in $(git branch -r --list 'upstream/*' | grep -v HEAD); do
        branch="${ref#upstream/}"
        echo "Syncing branch: $branch"
        if git show-ref --verify --quiet "refs/heads/$branch"; then
            git checkout "$branch"
            git merge "upstream/$branch" --no-edit
        else
            git checkout -b "$branch" "upstream/$branch"
        fi
    done
    # Sync tags
    echo "Syncing tags..."
    git fetch upstream --tags
else
    # Sync default branch only
    echo "Syncing branch: $UPSTREAM_DEFAULT"
    if git show-ref --verify --quiet "refs/heads/$UPSTREAM_DEFAULT"; then
        git checkout "$UPSTREAM_DEFAULT"
        git merge "upstream/$UPSTREAM_DEFAULT" --no-edit
    else
        git checkout -b "$UPSTREAM_DEFAULT" "upstream/$UPSTREAM_DEFAULT"
    fi
    git fetch upstream --tags
fi

echo "Pushing to origin..."
git push origin --all
git push origin --tags

echo "Sync complete."
