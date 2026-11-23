# Handoff

Running log for [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

## 2026-11-23: released 0.4.1

What changed: Released with `scripts/weirdnet_release.py` (approved by Michael, 2026-11-23). Release commit [`8681927`](https://github.com/WeirdBrains/weirdnet-backends/commit/8681927f0ef71feb5c953efa3a18a35c9aefedda), tag [`v0.4.1`](https://github.com/WeirdBrains/weirdnet-backends/releases/tag/v0.4.1), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.1) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.1). See the entry below and [CHANGELOG.md](CHANGELOG.md).

## 2026-11-23: live LoRA adapter changes and per-request scale (0.4.1)

What changed (see [CHANGELOG.md](CHANGELOG.md)): `Supervisor::add_adapter_live`/`remove_adapter_live` (vLLM run-time `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`; vLLM with LoRA on starts with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; `ModelLaunch::lora_slots` turns LoRA on with no adapters at start); `set_adapter_scale` (llama.cpp, live); `Request::adapter_scale` per request (weirdnet-agent 0.5.2) and `complete_scaled`; llama.cpp re-adding a file the engine has loaded is live; models/1 `ServeAdapter`, `UnserveAdapter`, `AdapterScale`. Additive, so a patch release.

Checks run: fmt; clippy `-D warnings` default and `--all-features`; tests `--all-features` (32 unit, 10 adapter, 2 node); `cargo doc -D warnings`. The mock engine now serves vLLM's run-time LoRA endpoints and checks `--max-loras`.

Not tested: real llama-server and vLLM. Checked against llama.cpp b12628 server source: llama-server cannot load a new adapter file at run time (`POST /lora-adapters` only rescales those from `--lora`), so a new file on llama.cpp still needs a base restart. vLLM's run-time endpoints follow its docs; not run on a GPU here.

What's left and next: weirdnet-alerts to accept 0.4; Ymir to use the live calls.


## 2026-11-22: weirdnet-model 0.5.2 from crates.io

What changed: `weirdnet-model = "0.5.2"` from crates.io; dropped the development `[patch.crates-io]` that pointed at weirdnet-model's lora-adapters branch. Ready to merge to main and release 0.4.0.

Checks run: after this change, on this checkout: fmt; clippy `-D warnings` with `--all-features` and `--no-default-features`; tests with `--all-features`, default and `--no-default-features`; `cargo doc -D warnings` with `--all-features`.

Not tested: as in the entry below.

What's left and next: merge to main and release 0.4.0 (approved).


## 2026-11-22: LoRA adapters (branch lora-adapters)

What changed (the [lora-adapters](https://github.com/WeirdBrains/weirdnet-backends/tree/lora-adapters) branch, not on main; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `LoraAdapter` and `ModelLaunch::adapters` (`ModelLaunch` is now `#[non_exhaustive]`), `with_adapters`, `served_ids`, `check_adapters`. llama.cpp: `--lora PATH` per adapter and `--lora-init-without-apply`; each request sets `"lora":[{id,scale}]` (a base with adapters sends `"lora":[]`). vLLM: `--enable-lora`, a single `--lora-modules` list (only the last flag counts in vLLM) and `--max-loras N`; a request names the adapter as `model`.
- `openai::Target`, `chat_body_for`, `completion_body_for`.
- Supervisor: adapters resolve to their base for chat, completion, state and capabilities; `load(adapter)` loads the base (restarting it when the adapter was added while it ran); `unload` and `embed` of an adapter are refused; `remove_model(base)` removes its adapters; `ModelStatus::base` and a row per adapter; `add_adapter` (`AdapterChange::Live`/`NextLoad`) and `remove_adapter`.
- models/1: `Pull::adapter_for`, `RepoFits::adapters`, `StoredEntry` covers adapters; architecture and base-model mismatches are refused.
- `Cargo.toml` takes `weirdnet-model` from git (`[patch.crates-io]`, its lora-adapters branch) for the unreleased adapter API, and `weirdnet-model` is no longer optional. This is why the work stays off main.

Checks run: `cargo fmt --check`; clippy `-D warnings` default, `--all-features` and `--no-default-features`; tests default, `--all-features` (32 unit, 9 adapter, 2 node) and `--no-default-features`; `cargo doc` with `-D warnings`. Engine-free: the mock engine parses `--lora`/`--lora-modules` and answers per adapter and scale; GGUF LoRA headers are written by the tests. `cargo package` was skipped (the git patch).

Not tested: a real llama.cpp or vLLM with adapters (checked against the llama.cpp b12590 server source: the per-request `lora` field, ids not listed get scale 0, an empty list uses the global scales, which `--lora-init-without-apply` sets to 0); a real Hub adapter pull.

What's left and next: once weirdnet-model 0.5.2 is published, set `weirdnet-model = "0.5.2"`, drop `[patch.crates-io]`, `cargo update -p weirdnet-model`, merge to main and release as 0.4.0 (breaking), after m3talsmith approves.

## 2026-11-21: release 0.3.0

What changed: `release: v0.3.0` with `scripts/weirdnet_release.py` (approved by m3talsmith together with [weirdnet-model](https://crates.io/crates/weirdnet-model) 0.5.1 and [weirdnet-alerts](https://crates.io/crates/weirdnet-alerts) 0.2.0, released in that order). Tag `v0.3.0`, published to [crates.io](https://crates.io/crates/weirdnet-backends/0.3.0), docs on [docs.rs](https://docs.rs/weirdnet-backends/0.3.0). Breaking for `node` users: weirdnet-model 0.5.1 is required and 0.4 is no longer accepted.

Checks run: CI green on main (6977f1f) before the release; the release script's checks from `release.toml`.

Not tested: as in the entries below.

What's left and next: weirdnet-site pages for backends 0.3.0; Ymir can move to weirdnet-backends 0.3 with weirdnet-model 0.5.1 and weirdnet-alerts 0.2.

## 2026-11-21: weirdnet-model 0.5.1 from crates.io, merged to main

What changed (merges the [ymir-models](https://github.com/WeirdBrains/weirdnet-backends/tree/ymir-models) branch into main; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `Cargo.toml` takes [weirdnet-model](https://crates.io/crates/weirdnet-model) `"0.5.1"` from crates.io and drops the `[patch.crates-io]` git patch; `Cargo.lock` follows. The `node` feature no longer takes weirdnet-model 0.4 (breaking, noted in the changelog).
- `models/1` and `Supervisor::add_model` are now on main.

Checks run: `cargo fmt --check`; clippy `-D warnings` default and `--all-features`; tests default, `--all-features` (28 unit, 8 adapter, 2 node) and `--no-default-features`; `cargo doc` with `-D warnings`; `cargo package`.

Not tested: as in the models/1 entry below (real transport, mid-download stream close, gated repos, real engine load after a pull).

What's left and next: release 0.3.0 (approved by m3talsmith together with weirdnet-model 0.5.1 and weirdnet-alerts 0.2.0).

## 2026-11-21: models/1 (branch ymir-models)

What changed (commit 61841af on the [ymir-models](https://github.com/WeirdBrains/weirdnet-backends/tree/ymir-models) branch, not on main; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `models` (feature `node`): `/weirdnet-backends/models/1`. `ModelsService` admits only what its `Authorize` admits (meant for `RosterAuth`, the owner's devices) and runs nothing for anyone else. `Search`, `Files` (fits against the node's budget, KV from the repo's GGUF header), `Pull` (progress, resume, split GGUFs, the largest that fits when no model is named; `serve`/`load` hand it to the supervisor through `add_model`), `List` and `Remove` (stops serving first). Closing a pull's stream cancels it. `ModelsServer` serves it on a `weirdnet-net` node; `ModelsClient` is the owner's side. The Hugging Face token is the node's own and is never in a message.
- `Cargo.toml` takes `weirdnet-model` from git (`[patch.crates-io]`, weirdnet-model main) because it needs the unreleased Hub, fit, token and split-download API. This is why the work stays off main.

Checks run: `cargo fmt --check`; clippy `-D warnings` default and `--all-features`; tests default, `--all-features` (28 unit, 8 adapter, 2 node) and `--no-default-features`; `cargo doc` with `-D warnings`. The models/1 tests run over a memory dialer against a local mock Hub: a stranger is refused before anything runs; the owner searches, sees fits, pulls the best fit and a two-part split model, serves without a restart, lists and removes. CI runs only on main, so this branch has local checks only. `cargo package` was skipped (the git patch).

Not tested: models/1 over a real `weirdnet-net` transport with `RosterAuth` (memory dialer only); cancelling by closing the stream mid-download (cancellation through the progress callback is tested); a real gated repo or real engine load after a pull.

What's left and next: once weirdnet-model 0.5.1 is published, set `weirdnet-model = "0.5.1"`, drop `[patch.crates-io]`, merge to main and release as 0.3.0 (breaking: drops weirdnet-model 0.4), after m3talsmith approves the versions.

## 2026-11-21: models at run time, weirdnet-model 0.5

What changed (commit cf294da, on main, unreleased; see [CHANGELOG.md](CHANGELOG.md) `Unreleased`):

- `Supervisor::add_model` and `Supervisor::remove_model` change the served models without a restart, with `SupervisorEvent::Added` and `Removed`; `Supervisor::launches` lists the launch settings.
- The `node` feature takes [weirdnet-model](https://crates.io/crates/weirdnet-model) `">=0.4, <0.6"`. Its `hw::Hardware` and `hw::Budget` are the same in 0.4 and 0.5, and an app on 0.5 now gets a single copy (it got two with 0.2.0).

Checks run: `cargo fmt --check`; clippy `-D warnings` default and `--all-features`; tests default, `--all-features` (26 unit, 8 adapter, 2 node) and `--no-default-features`; `cargo doc` with `-D warnings`; `cargo package`. A dependency probe with weirdnet-agent 0.5, weirdnet-model 0.5, weirdnet-mcp 0.5.3 and this crate resolves one weirdnet-agent and one weirdnet-model.

Not tested: a real engine with a model added at run time (the mock engine only); Windows.

What's left and next: a patch release (proposed 0.2.1) once m3talsmith approves it. The `models/1` protocol (search, fit, pull, list, remove for the owner) is on the [ymir-models](https://github.com/WeirdBrains/weirdnet-backends/tree/ymir-models) branch until weirdnet-model 0.5.1 is out.
