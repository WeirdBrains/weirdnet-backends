# Handoff

Running log for [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

## 2026-11-21: models at run time, weirdnet-model 0.5

What changed (commit cf294da, on main, unreleased; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `Supervisor::add_model` and `Supervisor::remove_model` change the served models without a restart, with `SupervisorEvent::Added` and `Removed`; `Supervisor::launches` lists the launch settings.
- The `node` feature takes [weirdnet-model](https://crates.io/crates/weirdnet-model) `">=0.4, <0.6"`. Its `hw::Hardware` and `hw::Budget` are the same in 0.4 and 0.5, and an app on 0.5 now gets a single copy (it got two with 0.2.0).

Checks run: `cargo fmt --check`; clippy `-D warnings` default and `--all-features`; tests default, `--all-features` (26 unit, 8 adapter, 2 node) and `--no-default-features`; `cargo doc` with `-D warnings`; `cargo package`. A dependency probe with weirdnet-agent 0.5, weirdnet-model 0.5, weirdnet-mcp 0.5.3 and this crate resolves one weirdnet-agent and one weirdnet-model.

Not tested: a real engine with a model added at run time (the mock engine only); Windows.

What's left and next: a patch release (proposed 0.2.1) once m3talsmith approves it. The `models/1` protocol (search, fit, pull, list, remove for the owner) is on the [ymir-models](https://github.com/WeirdBrains/weirdnet-backends/tree/ymir-models) branch until weirdnet-model 0.5.1 is out.
