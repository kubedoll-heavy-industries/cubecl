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

### 2026-06-08 — Add `cubecl-shim-gen` crate

**Upstream PR/issue**: not yet filed; will pitch to the team in
conversation rather than via GitHub.

**Why we carry**: build-time emitter for CubeCL CUDA kernels.
Produces `.cu` source + a Rust constants module per kernel,
intended to be consumed by build scripts of downstream crates
(`mistralrs-paged-attn` for paged KV cache scatter, `candle-cubecl`
for candle op ports). The crate sits naturally inside CubeCL —
it's a CubeCL-specific tool — so versioning stays in lockstep with
the emission semantics it depends on. Lives in
`crates/cubecl-shim-gen/`.

**Removal trigger**: equivalent functionality lands upstream, or
upstream rejects the contribution and we choose to keep maintaining
it as a fork-only crate.

**Commits**: `4bf0b58e..HEAD` (single commit so far).

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
