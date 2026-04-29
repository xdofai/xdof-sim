---
name: sync-upstream
description: 'Sync the latest code from an upstream remote into the current fork by cherry-picking each new upstream commit. Preserves fork-local files and changes. Use when the user says "sync upstream", "pull upstream", "update from upstream", or similar.'
argument-hint: "<upstream_remote/branch, default: upstream/main>"
---

# Sync Upstream

Sync the latest code from an upstream repository into the current fork by cherry-picking each new upstream commit onto the fork's branch.

This skill is built for **fork-by-copy repos** — forks created by copying a snapshot rather than by `git fork`, so `HEAD` and `<remote>/<branch>` share no merge base. We work at the **commit level**: identify the last synced upstream sha (the "anchor"), enumerate new upstream commits since then, and cherry-pick them in order. This preserves authorship, commit messages, and PR numbers, and surfaces conflicts in the natural context of a single change.

The previous version of this skill worked at the file level (snapshot the upstream tree, then re-apply fork-local patches). That approach was strictly more dangerous: a fork-local feature whose commits predated the last anchor could silently disappear (real incident: `9b6244d`). Cherry-pick avoids this because each upstream commit's diff is applied independently — fork-local code is only touched when an upstream commit overlaps it.

## Workflow

1. **Parse arguments**: default to `upstream/main` if none given. Split into `<remote>` and `<branch>`.

2. **Preflight**:
   - Working tree clean (`git status --porcelain`). If dirty, abort and tell the user to commit/stash first.
   - Upstream remote exists. If not, ask for the URL and add it.
   - Checkout the local branch matching `<branch>` (usually `main`).

3. **Fetch upstream**:
   ```bash
   git fetch <remote> <branch>
   ```

4. **Find the sync anchor** — the upstream sha most recently merged into the fork. Look for a commit whose message starts with `Sync upstream <remote>/<branch>@<sha>`:
   ```bash
   git log --grep="^Sync upstream ${remote}/${branch}@" -1 --format="%H %s"
   ```
   Parse the `<sha>` from the message.

   **If no anchor found** (first run on this fork): ask the user for the upstream sha the fork was originally copied from. Do NOT auto-guess — confirm before continuing.

5. **List new upstream commits**:
   ```bash
   git log --oneline --no-decorate <anchor>..<remote>/<branch>
   git log --merges --oneline <anchor>..<remote>/<branch>
   ```
   Show the user the list. Flag any merge commits — they need `-m 1` during cherry-pick.

   **If the range is empty**, exit: nothing to sync.

6. **Sanity-check the anchor**: verify the fork's HEAD tree differs from the anchor's tree only in fork-local files:
   ```bash
   git diff --stat <anchor> HEAD
   ```
   The output should be a small set of fork-local files (this is the **pre-sync fork-local delta**). If it's huge or includes obvious upstream files, the anchor is wrong — stop and ask the user. Save this delta — step 8 compares against it.

7. **Cherry-pick the range** in chronological order:
   ```bash
   git cherry-pick -x <anchor>..<remote>/<branch>
   ```
   - `-x` records the original upstream sha in each cherry-picked commit's message (`(cherry picked from commit ...)`) — preserves provenance.
   - For merge commits in the range, prefer cherry-picking each one with `-m 1` individually, or split the range so the merge commit gets its own invocation.

   **On conflict**: cherry-pick stops with `CHERRY_PICK_HEAD` set. Show the user:
   - Which commit failed (`git status`)
   - Which files conflict
   - The original commit's diff (`git show <orig_sha>`) so they can see intent

   Wait for resolution. After: `git add <files> && git cherry-pick --continue`. If the commit doesn't apply (already there, or not relevant): `git cherry-pick --skip`. If the whole sync should be aborted: `git cherry-pick --abort`.

   When cherry-pick auto-merges (no conflict markers), don't second-guess — the result is correct by definition. Move on.

8. **Verify fork-local deltas survived**. Compare HEAD to upstream's new tip:
   ```bash
   git diff --stat <remote>/<branch> HEAD -- ':!.claude*'
   ```
   The result should match the pre-sync fork-local delta from step 6 — same files, similar line counts. If a fork-local file shrank unexpectedly, a feature was clobbered — investigate before committing the anchor.

9. **Add the new anchor commit**:
   ```bash
   NEW_SHA=$(git rev-parse --short <remote>/<branch>)
   git commit --allow-empty -m "Sync upstream <remote>/<branch>@${NEW_SHA}"
   ```
   The empty commit is the anchor for the next sync. Do NOT auto-push. Ask the user.

10. **Summary**: list the cherry-picked commits, any conflicts that were resolved, the final fork-local delta vs upstream, and ask about pushing.

## Rules

- Always fetch before comparing.
- Cherry-pick in **chronological order** (oldest first) — `<anchor>..<remote>/<branch>` gives you this naturally.
- Always use `-x` so the original upstream sha is recorded in the cherry-picked commit. Provenance matters for future debugging.
- Use `-m 1` for merge commits in the range.
- Do NOT use three-dot diffs (`A...B`) — they require a merge base, which fork-by-copy repos lack.
- The anchor commit is the **source of truth** for "what's already been synced." Never skip step 6's sanity check — a wrong anchor either no-ops (too new) or replays already-synced commits (too old).
- If cherry-pick conflicts, resolve in the upstream commit's context. Each commit's diff is small and self-contained; conflicts usually mean a fork-local feature touches the same area as the upstream change.
- The anchor commit message format is **load-bearing** — `Sync upstream <remote>/<branch>@<sha>` exactly. Step 4's grep depends on it.
- Do NOT auto-push. Always ask first.
- Do NOT silently drop a fork-local file. Step 8's delta check is the safety net.

## Fallback: file-snapshot mode

Only use this if cherry-pick is structurally impossible — e.g. upstream did a history rewrite, or the anchor is unrecoverable and the user can't supply one. The fallback is strictly more dangerous (clobbers everything, relies on symbol verification to catch drops).

1. Classify files: `git diff --name-status HEAD <remote>/<branch>` (A=add, D=local-only preserve, M=candidate).
2. Compute "real conflict" files = M ∩ files-touched-by-non-sync-local-commits:
   ```bash
   git log --name-only --pretty=format: \
     --invert-grep --grep="^Sync upstream" --grep="^Initial commit" \
     | sort -u
   ```
3. For each real conflict, compute fork-local delta with `git diff <remote>/<branch> HEAD -- <f>` and confirm a re-apply plan with the user.
4. Extract distinctive symbols from non-sync commits per conflict file:
   ```bash
   for sha in $(git log --format=%H --invert-grep --grep="^Sync upstream" --grep="^Initial commit" -- <f>); do
     git show "$sha" -- <f> | grep '^+' | grep -oE '[A-Za-z_][A-Za-z0-9_]{8,}'
   done | sort -u > /tmp/fork_symbols_<f>.txt
   ```
5. Apply: `git checkout <remote>/<branch> -- .`
6. Re-apply each fork-local delta from step 3.
7. Verify every symbol from step 4 still exists in the file. Any missing symbol means a fork-local feature was dropped — re-apply before committing.
8. Commit: `git commit -m "Sync upstream <remote>/<branch>@<sha>"`.

The fallback's chief failure mode (which cherry-pick avoids): a fork-local feature added in a commit that predates the last sync may not appear in any "last sync → HEAD" diff if no later commit touched the same file, so the symbol-verification list is the only thing standing between you and a silent drop.
