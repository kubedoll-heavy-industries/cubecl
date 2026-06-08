# CubeCL fork divergence log

This is a soft fork of [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl).
We track upstream `main` and carry patches only when necessary.

See `~/Workspace/.claude/.../memory/decision_candle_on_cubecl.md` for
the strategic rationale.

## Remote configuration

```
upstream — https://github.com/tracel-ai/cubecl.git  (read-only)
origin   — git@github.com:Kubedoll-Heavy-Industries/cubecl.git
```

Branches:
- `main` — our fork's mainline; tracks `origin/main`
- `tracking` — clean mirror of `upstream/main`, never carries our changes

## Sync workflow

```bash
git fetch upstream
git checkout tracking
git reset --hard upstream/main
git push origin tracking

# Apply our patches on top of new upstream
git checkout main
git rebase tracking
git push origin main --force-with-lease
```

Weekly cron in `.github/workflows/sync.yml` (TODO) automates the
`tracking` branch update.

## Quarterly divergence review

Trigger: count of carried patches in `git log tracking..main`.

- <5 patches: stay soft, continue tracking
- 5–10: tighten upstream cadence, push back on stalled upstream PRs
- 10+: serious hard-fork consideration

## Carried patches

*(none yet — fork created 2026-06-08, tracking upstream as of `ba103c7f`)*

When a patch lands here that hasn't been accepted upstream, add an
entry below in this format:

```
### <date> — <one-line summary>

**Upstream PR/issue**: <URL or "not yet filed">

**Why we carry**: <one paragraph: what gap, why our roadmap needs it,
what blocks upstream from accepting it now>

**Removal trigger**: <upstream PR # merging, or "n/a — permanent fork
for project-specific reason">

**Commits**: `<sha>..<sha>`
```
