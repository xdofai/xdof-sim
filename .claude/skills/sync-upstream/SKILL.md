---
name: sync-upstream
description: 'Sync the latest code from an upstream remote into the current fork. Detects local-only changes and handles conflicts interactively. Use when the user says "sync upstream", "pull upstream", "update from upstream", or similar.'
argument-hint: "<upstream_remote/branch, default: upstream/main>"
---

# Sync Upstream

Sync the latest code from an upstream repository into the current fork's main branch, preserving any local-only files and changes.

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

4. **Analyze differences** between `HEAD` and `<remote>/<branch>`:
   - List files only in upstream (new upstream files)
   - List files only in local (local-only files)
   - List files modified on both sides (potential conflicts)
   - List files modified only on upstream side (safe to update)
   Use `git diff --name-status HEAD...<remote>/<branch>` and compare.

5. **Report diff summary** to the user before making any changes. Example:
   - N files updated from upstream (safe)
   - N new files from upstream
   - N local-only files (will be preserved)
   - N files with potential conflicts (need review)

6. **Apply changes**:
   - For safe updates and new upstream files: checkout from upstream directly
     ```bash
     git checkout <remote>/<branch> -- <file_path>
     ```
   - For local-only files: leave untouched
   - For conflict files: show the diff to the user for EACH file, then resolve:
     - If the local change is trivial (e.g., only in the fork's tooling/config), prefer upstream
     - If the local change is meaningful, attempt a merge and ask the user to confirm
     - Let the user decide when intent is ambiguous

7. **Stage and commit**:
   ```bash
   git add -A
   ```
   - Commit message format: `Sync upstream <remote>/<branch>@<short_sha>`
   - Do NOT auto-push. Ask the user if they want to push.

8. **Summary**: report what was synced, what was preserved, and what conflicts were resolved.

## Rules

- Always fetch before comparing to get the latest remote state.
- NEVER silently overwrite local-only changes. The whole point of this skill is to be safe.
- Preserve local-only files (files that don't exist in upstream).
- If there are no changes to sync, say so and exit.
- Do NOT use `git merge` or `git rebase` — this skill works at the file level to avoid unrelated-history issues common in fork-by-copy repos.
- Do NOT auto-push. Always ask first.
