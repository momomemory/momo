# Momo Release Strategy

> Decision document for how Momo manages releases, versioning, and artifact publishing.

## Artifacts Published Per Release

| Artifact            | Registry                  | Format                     |
| ------------------- | ------------------------- | -------------------------- |
| Linux x86_64 binary | GitHub Releases           | `momo-linux-amd64.tar.gz`  |
| macOS x86_64 binary | GitHub Releases           | `momo-darwin-amd64.tar.gz` |
| macOS ARM64 binary  | GitHub Releases           | `momo-darwin-arm64.tar.gz` |
| Docker image        | `ghcr.io/momomemory/momo` | OCI (linux/amd64)          |
| SHA256 checksums    | GitHub Releases           | `checksums.sha256`         |

---

## Options Considered

### Option 1: Tag-Driven Fully Automated Releases

**Trigger:** Push a semver tag (`v*.*.*`).

**Flow:**

```
git tag v0.2.0
git push origin v0.2.0
→ CI builds binaries + Docker image
→ Creates GitHub Release with assets
→ Done
```

**Pros:**

- Zero ceremony — tag and walk away.
- Deterministic: a tag always produces a release.
- Easy to automate from scripts or CI bots (e.g., release-please).

**Cons:**

- No gate between "I think this is ready" and "it's live." A bad tag means a bad release.
- Rollback = delete tag + delete release + push new tag. Messy.
- No place for human review of the release notes before publish.
- Accidental tags (typos, wrong branch) produce broken releases.

**Best for:** Solo maintainer projects with strong CI coverage and few consumers.

---

### Option 2: Manual Dispatch with Inputs

**Trigger:** `workflow_dispatch` with version input.

**Flow:**

```
GitHub Actions UI → "Run workflow"
→ Enter version: 0.2.0
→ Optionally toggle "dry run" or "pre-release"
→ CI builds everything, creates release
```

**Pros:**

- Full human control over timing and version.
- Can add dry-run mode, pre-release flags, target branch selection.
- No accidental releases from stray tags.

**Cons:**

- Requires visiting the GitHub Actions UI (or `gh workflow run` CLI).
- Easy to forget steps or mis-type the version.
- Version in `Cargo.toml` can drift from the dispatch input.
- Doesn't scale well — every release is a manual ceremony.

**Best for:** Teams that want explicit approval gates or infrequent releases.

---

### Option 3: Hybrid — Tag-Triggered with Manual Override ✅ RECOMMENDED

**Triggers:**

1. Push a semver tag (`v*.*.*`) → full automated release.
2. `workflow_dispatch` with optional version override → same pipeline, manual trigger.

**Flow (typical):**

```
# Automated path (day-to-day)
git tag v0.2.0 && git push origin v0.2.0
→ Full release pipeline runs

# Manual path (hotfix, re-release, dry run)
gh workflow run release.yml -f version=0.2.1 -f dry_run=true
→ Same pipeline, human-initiated
```

**Pros:**

- Fast path for routine releases (just tag).
- Escape hatch for edge cases (manual dispatch).
- Single workflow file handles both paths — no duplication.
- `workflow_dispatch` enables dry runs, pre-release toggles, and testing the pipeline itself.
- Tag-based triggers integrate cleanly with tools like `release-please`, `cargo-release`, or custom scripts.

**Cons:**

- Slightly more complex workflow YAML (branching on trigger type).
- Must document both paths so contributors know the options.

**Best for:** Projects expecting growth, multiple contributors, and long-term maintenance.

---

## Recommendation: Option 3 (Hybrid)

**Why this wins for Momo:**

1. **Scales with the project.** Today it's a solo maintainer tagging releases. Tomorrow it could be a team with a release manager reviewing changelogs before publish. The hybrid approach supports both without workflow changes.

2. **Testable pipeline.** The `workflow_dispatch` path with `dry_run: true` lets you validate the entire release pipeline without publishing anything. This is critical for a project with complex native deps (Tesseract, Whisper, tree-sitter).

3. **Compatible with automation tools.** When the project grows, adding `release-please` or `cargo-release` is trivial — they create tags, which trigger the existing workflow. No migration needed.

4. **Graceful error recovery.** If a tag-triggered release fails mid-way, you can re-run it manually via dispatch without deleting and re-creating the tag.

5. **Pre-release support.** Tags like `v0.3.0-rc.1` can automatically be marked as pre-releases, giving early adopters access without affecting the `latest` tag.

---

## Versioning Policy

- Follow [Semantic Versioning 2.0](https://semver.org/).
- Source of truth: `Cargo.toml` `version` field.
- Tags MUST match the Cargo.toml version (the workflow validates this).
- Pre-release tags (`-alpha`, `-beta`, `-rc.N`) are supported and auto-detected.
- Docker tag `latest` only updates on stable (non-pre-release) tags.

## Future Enhancements

These are explicitly **not** in the initial workflow but are easy to add later:

| Enhancement                               | Effort | When to Add                         |
| ----------------------------------------- | ------ | ----------------------------------- |
| `release-please` for automated changelogs | Low    | When commit volume justifies it     |
| `cargo-release` for version bumps         | Low    | When manual bumps become tedious    |
| Linux ARM64 binary                        | Medium | When there's user demand            |
| Windows binary                            | Medium | When there's user demand            |
| Crates.io publish                         | Low    | If/when the crate becomes a library |
| Cosign/Sigstore signing                   | Low    | When supply-chain security matters  |
| SBOM generation                           | Medium | For enterprise/compliance users     |

---

_Last updated: 2026-02-08_
