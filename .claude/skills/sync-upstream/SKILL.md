---
name: sync-upstream
description: 'Sync the latest code from an upstream remote into the current fork. Detects local-only changes and handles conflicts interactively. Use when the user says "sync upstream", "pull upstream", "update from upstream", or similar.'
argument-hint: "<upstream_remote/branch, default: upstream/main>"
---

# Sync Upstream

Sync the latest code from an upstream repository into the current fork's main branch, preserving any local-only files and changes.

This skill is built for **fork-by-copy repos** — forks created by copying a snapshot rather than by `git fork`, so `HEAD` and `<remote>/<branch>` share no merge base. All commands below use two-arg (tree-to-tree) diffs, never three-dot.

## Workflow

1. **Parse arguments**: default to `upstream/main` if no argument given. Split into `<remote>` and `<branch>`.

2. **Preflight checks**:
   - Ensure working tree is clean (`git status --porcelain`). If dirty, abort and tell the user to commit or stash first.
   - Ensure the upstream remote exists. If not, ask the user for the upstream URL and add it.
   - Checkout the local branch that corresponds to the upstream branch (usually `main`).

3. **Fetch upstream**:
   ```bash
   git fetch <remote> <branch>
   ```

4. **Classify every file** between `HEAD` and `<remote>/<branch>`:
   ```bash
   git diff --name-status HEAD <remote>/<branch>
   ```
   Do NOT use three-dot (`HEAD...<remote>/<branch>`) — it requires a merge base and will fail with `fatal: no merge base` on fork-by-copy repos.

   Status codes:
   - `A` — exists in upstream, not in HEAD → new upstream file (add)
   - `D` — exists in HEAD, not in upstream → local-only file (preserve, never touch)
   - `M` — exists on both sides with different content → *candidate* for conflict, see step 5

5. **Separate real conflicts from drift**. In fork-by-copy repos, most `M` files are drift from prior file-level syncs, not real local edits. Only files touched by non-sync commits are real conflicts.

   Enumerate files ever touched by a real local commit in one call:
   ```bash
   git log --name-only --pretty=format: \
     --invert-grep --grep="^Sync upstream" --grep="^Initial commit" \
     | sort -u
   ```
   (Extend the `--grep` list if the repo uses other auto-sync commit prefixes.)

   Then partition the `M` set:
   - **Real conflicts** = M ∩ files-with-real-local-edits
   - **Drift (safe overwrite)** = M \ files-with-real-local-edits

   Never loop `git log` over each `M` file individually — the single batch command above is the only classification call needed.

6. **Report diff summary** to the user before making any changes:
   - N new files from upstream (safe)
   - N drift-only files (safe overwrite)
   - N local-only files (will be preserved)
   - N real conflicts (need review) — list these by name

7. **Show each real conflict as a clean local delta**, not a pairwise HEAD↔upstream diff (which includes all unrelated upstream churn). For each real-conflict file `<f>`:
   ```bash
   LAST_SYNC=$(git log --format=%H --grep="^Sync upstream" -- <f> | head -1)
   git diff "$LAST_SYNC" HEAD -- <f>
   ```
   This is the pure local delta — what was added on top of the last-synced upstream version. It's what needs to be re-applied after overwriting with upstream.

   For each conflict, propose a resolution to the user (most commonly: "apply upstream's version, then re-append these N lines at `<anchor>`") and wait for confirmation.

8. **Apply**:
   ```bash
   git checkout <remote>/<branch> -- .
   ```
   This overwrites every file present in upstream's tree in one shot — both safe updates and drift. Local-only files are untouched because they don't exist in upstream. Then re-apply each confirmed local delta from step 7.

9. **Stage and commit**:
   ```bash
   git add -A
   git commit -m "Sync upstream <remote>/<branch>@<short_sha>"
   ```
   Do NOT auto-push. Ask the user if they want to push.

10. **Summary**: what was synced, what was preserved, what conflicts were resolved, and any caveats (e.g. anchor-dependent patches that should be runtime-verified).

## Rules

- Always fetch before comparing to get the latest remote state.
- NEVER silently overwrite local-only changes. The whole point of this skill is to be safe.
- Preserve local-only files (files that don't exist in upstream).
- If there are no changes to sync, say so and exit.
- Do NOT use `git merge` or `git rebase` — this skill works at the file level to avoid unrelated-history issues common in fork-by-copy repos.
- Do NOT use three-dot diffs (`A...B`) anywhere — they require a merge base.
- In fork-by-copy repos, `M` does NOT mean conflict. Classify `M` as *drift* vs *real edit* using the batch `git log --invert-grep` command in step 5. Only show diffs and ask the user about *real edits*.
- Do NOT auto-push. Always ask first.
