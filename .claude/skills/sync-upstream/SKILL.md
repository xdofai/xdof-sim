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
   (Extend the `--grep` list if the repo uses other auto-sync commit prefixes. Do NOT pipe through `head` — enumerate the full set.)

   Then partition the `M` set:
   - **Real conflicts** = M ∩ files-with-real-local-edits
   - **Drift (safe overwrite)** = M \ files-with-real-local-edits

   Never loop `git log` over each `M` file individually — the single batch command above is the only classification call needed.

6. **Report diff summary** to the user before making any changes:
   - N new files from upstream (safe)
   - N drift-only files (safe overwrite)
   - N local-only files (will be preserved)
   - N real conflicts (need review) — list these by name

   **Branch point**:
   - **Zero real conflicts** → proceed end-to-end without pausing (skip step 7 entirely; still run steps 8–10 for any files that intersect `M` ∩ real-local-edits, which by definition is empty here; go to step 11 and commit). Report the final summary afterwards.
   - **One or more real conflicts** → continue to step 7 and **wait for user confirmation** on the resolution plan before applying anything.

7. **Compute the full fork-local delta per conflict file**. For each real-conflict file `<f>`:
   ```bash
   git diff <remote>/<branch> HEAD -- <f>
   ```
   Read the hunks as:
   - `+` lines = in HEAD, missing from upstream → **fork-local; must preserve**
   - `-` lines = in upstream, missing from HEAD → upstream additions we'll gain after checkout

   This diff shows the **total** fork-local delta, not just what was added since the last sync. That matters because fork-local features can be introduced BEFORE a later sync commit that didn't happen to touch the file, and a "last sync → HEAD" diff will silently miss them. (Real incident: the joystick feature in `vr_streamer.py` was added in commits predating the last sync touching that file, so `git diff LAST_SYNC HEAD -- <f>` reported only the newer recording-indicator delta. The joystick block was dropped on the next sync. Post-mortem commit: `9b6244d`.)

   Enumerate each local hunk to the user and propose a resolution ("apply upstream, then re-append these N lines at `<anchor>`") — wait for confirmation before moving on.

8. **Collect fork-local symbols for post-checkout verification**. For each real-conflict file, extract distinctive symbols introduced by non-sync local commits:
   ```bash
   for sha in $(git log --format=%H \
                 --invert-grep --grep="^Sync upstream" --grep="^Initial commit" \
                 -- <f>); do
     git show "$sha" -- <f> | grep '^+' | grep -oE '[A-Za-z_][A-Za-z0-9_]{8,}'
   done | sort -u > /tmp/fork_symbols_<f>.txt
   ```
   Tune the regex per language if needed (e.g. lower the min-length for short Python identifiers). Keep the list short and distinctive — function/class/flag names, not boilerplate.

9. **Apply**:
   ```bash
   git checkout <remote>/<branch> -- .
   ```
   This overwrites every file present in upstream's tree in one shot — both safe updates and drift. Local-only files are untouched because they don't exist in upstream.

10. **Re-apply local deltas + verify**. Re-apply each confirmed patch from step 7 onto the fresh upstream file. Then for every real-conflict file `<f>`:
    ```bash
    missing=$(while read -r sym; do grep -qF -- "$sym" <f> || echo "$sym"; done < /tmp/fork_symbols_<f>.txt)
    [ -n "$missing" ] && echo "MISSING in <f>: $missing"
    ```
    Any missing symbol means a fork-local piece of code did not survive — stop and re-apply it before continuing. Do NOT commit until this check is clean for every real-conflict file.

11. **Stage and commit** (no user confirmation needed — the branch point in step 6 already gated this on conflict presence):
    ```bash
    git add -A
    git commit -m "Sync upstream <remote>/<branch>@<short_sha>"
    ```
    Do NOT auto-push. Ask the user if they want to push.

12. **Summary**: what was synced, what was preserved, what conflicts were resolved, the symbol-verification result, and any caveats (e.g. anchor-dependent patches that should be runtime-verified).

## Rules

- Always fetch before comparing to get the latest remote state.
- NEVER silently overwrite local-only changes. The whole point of this skill is to be safe.
- Preserve local-only files (files that don't exist in upstream).
- If there are no changes to sync, say so and exit.
- Do NOT use `git merge` or `git rebase` — this skill works at the file level to avoid unrelated-history issues common in fork-by-copy repos.
- Do NOT use three-dot diffs (`A...B`) anywhere — they require a merge base.
- In fork-by-copy repos, `M` does NOT mean conflict. Classify `M` as *drift* vs *real edit* using the batch `git log --invert-grep` command in step 5. Only show diffs and ask the user about *real edits*.
- **"Last sync commit" is NOT a per-file merge base.** A sync commit that didn't modify file `<f>` leaves older fork-local content in `<f>` untracked by any subsequent baseline. Always compute the fork-local delta as `git diff <remote>/<branch> HEAD -- <f>` (full pairwise against upstream), never as `git diff LAST_SYNC HEAD -- <f>`.
- Every real-conflict file must pass the symbol-verification step (step 10) before commit. If a fork-local symbol is missing post-checkout, re-apply manually — do not proceed.
- **Confirmation is gated on real conflicts, not on the sync itself.** If step 5 finds zero real-conflict files, run through apply → verify → commit without pausing — the diff summary in step 6 is informational, not a prompt. Only wait for user confirmation when step 7's fork-local delta needs a resolution plan.
- Do NOT auto-push. Always ask first.
