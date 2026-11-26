# Handoff

Running log for [weirdnet-backends](https://github.com/WeirdBrains/weirdnet-backends), newest first. Shared rules: the collective `AGENTS.md` and this repo's [AGENTS.md](AGENTS.md).

## 2026-11-26: released 0.4.7 (upload switch, StoredEntry constructors)

What changed: released with `scripts/weirdnet_release.py --patch` from [`0c64ca5`](https://github.com/WeirdBrains/weirdnet-backends/commit/0c64ca5f6b40a7a40b73faee4724920dc7eb011b) (CI green, Linux and Windows). Release commit [`6c4eacd`](https://github.com/WeirdBrains/weirdnet-backends/commit/6c4eacdecfbc359b6ab826a03beb8845cb54eedd) (CI green), tag [`v0.4.7`](https://github.com/WeirdBrains/weirdnet-backends/tree/v0.4.7), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.7) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.7).

Checks run: the release script's checks and `cargo publish` verification. The `weirdnet-sync` requirement is already `>=0.5, <0.9` and `weirdnet-agent = "0.5.5"` takes 0.5.6. A scratch app with weirdnet-backends 0.4.7 (`node`), [weirdnet-agent](https://crates.io/crates/weirdnet-agent/0.5.6) 0.5.6 (`pool`, `serve`, `jobs`), weirdnet-sync 0.8.2 and weirdnet-net `=0.8.2` builds, with one weirdnet-sync (0.8.2) in the tree.

Not tested: no live nodes; macOS.

What's left and next: nothing for this change.

## 2026-11-26: upload switch and StoredEntry constructors (for 0.4.7)

What changed: two additive requests from the [Ymir](https://github.com/WeirdBrains/ymir) upload work.
- `WritePolicy::uploads` (default `true`) with `WritePolicy::with_uploads(bool)`: when off, `ModelsCommand::Upload` is refused with `refusal::NOT_ALLOWED` ("uploads are off on this node") before anything is staged, while pulls (Hub and `Pull::from_peer`), `Remove` and `AbandonUpload` keep working. Before, only `auth` (all writes) or `max_upload = Some(0)` (reads as `too-big`) could stop uploads.
- `StoredEntry::from_model`, `from_adapter`, `from_projector`, `From<StoredModel | StoredAdapter | StoredProjector>`, `with_served(served_as, state)`, `with_base(base)` and `StoredEntry::list(&ModelStore)`, which lists a store as entries (not served) in the order `ModelsService::list` uses. `ModelsService::list` now builds on it. Callers no longer need serde to build entries.
Commit SHA: see the commit that adds this entry.

Checks run: `cargo fmt --check`; `cargo clippy --all-targets --all-features -D warnings`; `cargo test` (default) and `cargo test --all-features` (47 unit tests with the new `uploads_can_be_turned_off_alone` and `entries_are_built_and_listed_without_serde`); `RUSTDOCFLAGS=-D warnings cargo doc --no-deps --all-features`; `cargo semver-checks --all-features` against 0.4.6.

Not tested: no live nodes; macOS (CI covers Linux and Windows).

What's left and next: release 0.4.7.

## 2026-11-25: released 0.4.6 (weirdnet-sync 0.8)

What changed: released with `scripts/weirdnet_release.py --patch` from [`fca59b6`](https://github.com/WeirdBrains/weirdnet-backends/commit/fca59b6ef455b7c402b723a11d9d3f048b106557) (CI green, Linux and Windows). Release commit [`e1e5e06`](https://github.com/WeirdBrains/weirdnet-backends/commit/e1e5e0689ffe268e3d344753d5c1407e147c4cbe) (CI green), tag [`v0.4.6`](https://github.com/WeirdBrains/weirdnet-backends/tree/v0.4.6), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.6) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.6).

Release basis: m3talsmith's 2026-11-25 decision to accept weirdnet-sync 0.8 here, under his standing rule of 2026-11-24 for additive Weirdnet patch releases. `cargo semver-checks --all-features` against 0.4.5: no semver update required.

Checks run: CI (including the sync 0.5.3 and 0.7.1 steps), the release script's checks and `cargo publish` verification. A scratch app using [weirdnet-openai](https://crates.io/crates/weirdnet-openai) 0.2.3, [weirdnet-semcache](https://crates.io/crates/weirdnet-semcache) 0.2.1, [weirdnet-alerts](https://crates.io/crates/weirdnet-alerts) 0.2.2 (`backends`), [weirdnet-model](https://crates.io/crates/weirdnet-model) 0.5.3 (`agent`), [weirdnet-mcp](https://crates.io/crates/weirdnet-mcp) 0.6.1 (`agent`), agent 0.5.5 and this release resolves to one weirdnet-sync 0.8.0 and builds.

Note for apps that stay on sync 0.7: a fresh resolve gives this crate 0.8.0 next to the app's 0.7.x (Cargo picks the newest version in a range across 0.x lines). Unify with `cargo update -p weirdnet-sync@0.8.0 --precise 0.7.1`; checked in the same scratch app.

Not tested: no live nodes on mixed sync versions.

What's left and next: nothing for this change.

## 2026-11-25: accept weirdnet-sync 0.8 (for 0.4.6)

What changed: the `weirdnet-sync` requirement (the `node` feature and the dev-dependency) widens from `>=0.5, <0.8` to `>=0.5, <0.9`, and `weirdnet-agent` goes from `0.5.2` to `0.5.5`, the first [weirdnet-agent](https://crates.io/crates/weirdnet-agent/0.5.5) that accepts sync 0.8 too. No code changes were needed. Cargo.lock now holds weirdnet-sync 0.8.0, weirdnet-net 0.8.2, weirdnet-keys 0.1.2 and weirdnet-agent 0.5.5 (the yanked sync 0.7.0, net 0.8.0 and keys 0.1.1 are gone). CI also tests sync 0.5.3 and 0.7.1. Commit SHA: see the commit that adds this entry.

Basis: m3talsmith decided on 2026-11-25 to widen the weirdnet-sync range in weirdnet-agent and weirdnet-backends to include 0.8, and his standing rule of 2026-11-24 covers additive Weirdnet patch releases. This is additive, so it ships as 0.4.6.

Checks run: `cargo fmt --check`; `cargo clippy --all-targets --all-features -D warnings` and `cargo test --all-features` against weirdnet-sync 0.8.0, 0.7.1 and 0.5.3 (the oldest version in range that is not yanked); `cargo test` with default and no default features; `cargo doc -D warnings`.

Not tested: sync 0.5.0 to 0.5.2 (yanked); no live nodes on mixed sync versions; Windows runs in CI only.

What's left and next: release 0.4.6.

## 2026-11-25: released 0.4.5

What changed: `v0.4.5` is out (release commit [`42e4256`](https://github.com/WeirdBrains/weirdnet-backends/commit/42e4256600426de30b3c3514030f3093b43fea1b), tag [v0.4.5](https://github.com/WeirdBrains/weirdnet-backends/releases/tag/v0.4.5), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.5) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.5)). It ships the placement hooks ([`006922b`](https://github.com/WeirdBrains/weirdnet-backends/commit/006922b)) and model origin and license in capability records (merge [`7a543ab`](https://github.com/WeirdBrains/weirdnet-backends/commit/7a543ab)), both below. The release ran from the origin and license work; a second release run for the hooks lost the atomic push to it and published nothing. This entry records the release.

Release approval: m3talsmith's standing approval of additive Weirdnet patch releases once main is green (2026-11-24, 9:21 PM ET).

Checks run: CI green on `7a543ab`; `cargo semver-checks` against 0.4.4 on `7a543ab` (no semver update required); the release script's checks.

Not tested: as in the entries below.

What's left and next: Ymir moves to 0.4.5, replaces its own uploads protocol with `on_placed` and `with_node_info`, and reads origin and license from capability records.

## 2026-11-25: model origin and license in capability records (for 0.4.5)

What changed ([`a7d406a`](https://github.com/WeirdBrains/weirdnet-backends/commit/a7d406a7c6dfc072ce45bb4b42859c7addba3d28), branch `feat/model-origin`, merged to main with `--no-ff`): `ModelEntry::origin` and `ModelEntry::license` (`Option<String>`, serde default, left out of the record when unset) and `NodeService::set_model_origin(model, origin, license)`, which keeps the revision and quantization from `set_model_info` (whose signature is unchanged) and pushes a fresh record. Origins are the [weirdnet-model](https://crates.io/crates/weirdnet-model) 0.5.3 provenance ones (`hub`, `upload`, `train`), and licenses are SPDX ids or `unknown`. Asked for by [Ymir](https://github.com/WeirdBrains/ymir) so its catalog ([weirdnet-agent](https://crates.io/crates/weirdnet-agent) 0.5.4 `CatalogModel::origin`/`license`) can say where a model came from. `ModelEntry` is `#[non_exhaustive]`, so this is additive: a patch release, 0.4.5.

After merging main's placement hooks ([`006922b`](https://github.com/WeirdBrains/weirdnet-backends/commit/006922b)): `ModelsService::with_node_info` also calls `set_model_origin` with each placed entry's origin and license (`Placed::origin_and_license`: the provenance's, else `upload` or `hub` by how it arrived; a peer copy without provenance has none). Both ship together as 0.4.5.

Checks run: `cargo fmt --check`, `cargo build`, `cargo clippy --all-targets` with and without `--all-features` (`-D warnings`), `cargo test --all-features` (45 unit, 12 adapters, 2 node; `node_info_follows_placements` now checks the uploaded origin and license reach the record; the node/1 round trip now checks that records without an origin look as before, that `set_model_origin` reaches the signed record with the revision kept, that a reader shaped like a 0.4.4 record still reads it, and that `set_model_info` keeps the origin), `cargo doc --no-deps --all-features` with `-D warnings`, and `cargo semver-checks` against 0.4.4 (no semver update required).

Not tested: no live model node; Hub pulls and peer copies through `with_node_info` (same hook as uploads).

Release basis: m3talsmith's standing rule of 2026-11-24 allows additive Weirdnet patch releases without asking; minor bumps still need his OK.

What's left and next: release 0.4.5 once CI on main is green; Ymir copies `ModelEntry::origin`/`license` into its catalog entries (weirdnet-agent 0.5.4 `CatalogModel::with_origin`/`with_license`), and calls `set_model_origin` itself for models it configures outside models/1.
## 2026-11-25: placement hooks on ModelsService (0.4.5)

What changed (see [CHANGELOG.md](CHANGELOG.md) `Unreleased` and the README section "Uploading your own models"), asked for by Ymir so apps no longer need their own protocol to record uploads:

- `models/placed.rs`: `ModelsService::on_placed(impl Fn(&Placed) + Send + Sync + 'static)` (several hooks, run in order; `OnPlaced` is the stored `Arc<dyn Fn(&Placed) + Send + Sync>`). `Placed { entry: StoredEntry, pulled: Pulled, via: PlacedVia, serve_error: Option<RemoteError> }` with `id()`, `revision()`, `provenance()`, `served_as()` and `quant()`; `PlacedVia::{Upload, Peer { node }, Hub { repo }}`. Both are `#[non_exhaustive]`.
- Uploads and peer copies (so each `ModelsClient::replicate` target) fire once the files are imported, after serving, and also when serving then fails (`serve_error` set; the client still gets the error). Hub model and adapter pulls fire on success. Serving a `Fetch` places nothing and fires nothing. Hooks run before the client's answer; a panicking hook is caught.
- `ModelsService::with_node_info(Arc<NodeService>)`: a hook that calls `NodeService::set_model_info(served id or store id, revision, quant from a GGUF model's file name)` for entries with a revision; Hub pulls (no revision) leave the record alone. It uses this crate's own `NodeService`, so no new dependency, and does not depend on origin or license in weirdnet-agent's `set_model_info`.
- `finish` in `models/upload.rs` now serves through a separate `serve_imported`, so a placement whose serving fails is still reported.

Release approval: m3talsmith's standing approval (2026-11-24, 9:21 PM ET) covers additive Weirdnet patch releases once main is green; this ships as 0.4.5.

Checks run: the `release.toml` checks (fmt; clippy `-D warnings` and tests with `--all-features`; `cargo doc -D warnings`); `cargo semver-checks` against 0.4.4. New tests: hooks on an upload (entry, provenance, revision, quant, served id) and on the peer copy made by `replicate`, with a panicking hook alongside; adapter and projector placements; a projector placed on an embedding model, where serving fails and `serve_error` matches the client's error; Hub model pulls (served and not); `with_node_info` putting the revision and quantization in the capability record.

Not tested: hooks on a real `weirdnet-net` node and with Ymir's app; a Hub adapter pull's hook (same code path as the model pull).

What's left and next: release 0.4.5; Ymir can drop `/ymir/uploads/1` in favour of `on_placed` and `with_node_info`. A projector for an embedding or wrapped llama.cpp model passes the upload's early checks and only fails when served.

## 2026-11-24: release 0.4.4

What changed: release `v0.4.4` (tag [v0.4.4](https://github.com/WeirdBrains/weirdnet-backends/releases/tag/v0.4.4), the `release: v0.4.4` commit right after this entry). It ships the models/1 upload work below (`eecf197`) on weirdnet-model 0.5.3 to [crates.io](https://crates.io/crates/weirdnet-backends/0.4.4); docs at [docs.rs](https://docs.rs/weirdnet-backends/0.4.4).

Release approval: m3talsmith approved publishing 0.4.4 to crates.io once main is green (2026-11-24, 9:21 PM ET).

Checks run: CI green on `eecf197`; the release script's checks from [release.toml](release.toml) (fmt, clippy `-D warnings` and tests with `--all-features`, `cargo doc -D warnings`), then `cargo publish`.

Not tested: as in the entry below.

What's left and next: Ymir's app moves to weirdnet-backends 0.4.4 and weirdnet-model 0.5.3 for uploads.

## 2026-11-24: model uploads, peer copies and write policy in models/1 (0.4.4)

What changed (see [CHANGELOG.md](CHANGELOG.md) `Unreleased` and the README section "Uploading your own models"), the node side of the [model upload plan](https://github.com/WeirdBrains/ymir/blob/feature/model-upload-plan/docs/design/model-upload.md), on [weirdnet-model 0.5.3](https://crates.io/crates/weirdnet-model/0.5.3) from crates.io:

- `models/upload.rs`: `ModelsCommand::Upload` (`Upload` → `UploadReady` → `UploadFrame::Chunk` of at most 48 KiB → `Done` → `Uploaded`), resumable from parts staged outside the store (`with_staging`, default `<store>.uploads`; swept after `WritePolicy::part_ttl`, 7 days), `AbandonUpload`. Checks before bytes move (name, size, quota, free space, engine, base), on early bytes (GGUF and safetensors headers, at doubling thresholds; adapters against their base's architecture), and on `Done` (every SHA-256), then `ModelStore::import_files` and serving: models by layout with `add_model` when the node runs that engine, adapters with `add_adapter_live`, projectors with the new `Supervisor::set_projector_live`. The node always records the caller as `uploaded_by`.
- `ModelsCommand::Fetch`, `PullSource::Peer` / `Pull::from_peer` (`with_peer_dialer`) and `ModelsClient::replicate` for node-to-node copies, resuming like uploads; the provenance travels with the files.
- `WritePolicy` with its own `Authorize`, `max_upload`, `quota` and `KeepFree` (default the larger of 10% and 20 GiB); `Remove` follows it too. Refusals are `RemoteError::Refused` with a `reason: details` message (`refusal`, `refusal_reason`), so weirdnet-agent needed no change. `ModelsCommand::Disk` / `DiskReport`.
- `List`: provenance on every entry, safetensors and MLX models, projectors (`StoredEntry::projector`, `base` set to the model using it). `Pulled::revision` and `projector_for`.
- `ModelLaunch::mmproj` (`with_mmproj`): llama.cpp gets `--mmproj`, `capabilities()` reports vision, and a served model restarts on its next load when its projector changes.

Decisions: m3talsmith chose that any linked device may upload, so the default `WritePolicy` lets in any caller the service's `Authorize` admits (any linked or rostered device under `RosterAuth`); no any-of-roles helper was added to `RosterAuth` (weirdnet-sync device roles are already free-form app strings, and `with_auth` can check one). A second offer of the same files while one is in flight gets `Busy` rather than joining the first. The node does not set the capability record's revision itself: the app passes `Pulled::revision` to `NodeService::set_model_info`.

Release approval: m3talsmith approved publishing 0.4.4 to crates.io once main is green (2026-11-24, 9:21 PM ET), after weirdnet-model 0.5.3 ([v0.5.3](https://github.com/WeirdBrains/weirdnet-model/releases/tag/v0.5.3), published 2026-11-24).

Checks run: the `release.toml` checks (fmt; clippy `-D warnings` and tests with `--all-features`; `cargo doc -D warnings`); `cargo semver-checks` against 0.4.3. New tests over `MemoryDialer`: upload, resume after a dropped stream, serve and provenance in `List`; every refusal reason the node can reach (including a wrong hash and a write `Authorize`); early wrong-architecture refusal of an adapter; adapter and projector uploads served live; a safetensors model served on vLLM; a copy between two services with `replicate`; stale and abandoned parts; `--mmproj` and vision in capabilities.

Not tested: real engines with uploaded files (a real llama-server with `--mmproj`, vLLM with an uploaded safetensors folder, MLX-LM); uploads over a real `weirdnet-net` link and multi-GB files; disk-full during an upload; `no-space` through the network path (tested on the service).

What's left and next: release 0.4.4; Ymir's app wires the upload UI, `set_model_info` with `Pulled::revision`, and its write policy.

## 2026-11-24: released 0.4.3

What changed: Released with `scripts/weirdnet_release.py` (the reserve command was approved by Michael, 2026-11-24). Release commit [`cbd2a19`](https://github.com/WeirdBrains/weirdnet-backends/commit/cbd2a19), tag [`v0.4.3`](https://github.com/WeirdBrains/weirdnet-backends/releases/tag/v0.4.3), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.3) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.3). See the entry below and [CHANGELOG.md](CHANGELOG.md). Next: Ymir P3a uses it.

## 2026-11-24: node reservations (0.4.3)

What changed (see [CHANGELOG.md](CHANGELOG.md)): Michael approved a "hold this GPU, place nothing here" command for Ymir's training plan (P3a), 2026-11-24. `Supervisor::reserve(holder)` sets the reservation first, so loads waiting on `ops` see it. It then stops every running engine under `ops` (each drains up to `drain_timeout`). `load_with` refuses while reserved (checked before the backoff wait and again under `ops`; `force` too), so loads on demand, swaps and crash restarts start nothing. `ModelHost` refuses requests (with `wait_for_model` too). The same holder again is a no-op; another holder gets `Unavailable`; `release` by another holder gets `BadRequest`. The reservation is in memory only: an application that must keep it across a restart records it and reserves again at start (Ymir does). node/1: `Reserve { holder }`, `Release { holder }`, `NodeClient::reserve`/`release`, `Capability::reserved`, `Health::reserved` (serde default, skipped when none, so records stay the same for unreserved nodes and older readers), and `GroupUp` refused while reserved. `NodeService::watch` pushes a fresh record on every reservation change. There are no new `SupervisorEvent` variants, because older readers of `Health::events` would fail on them. Additive, so a patch release.

Checks run: fmt; clippy `--all-features --all-targets -D warnings`; tests `--all-features` (33 unit, 12 adapter, 2 node); docs with `-D warnings`. New: `a_reservation_stops_every_engine_and_starts_none`, and a reserve section in `node1_round_trip` (record, refused load, infer and group job, another holder, health, release).

Not tested: real engines; a node before 0.4.3 receiving `Reserve` (it can't decode the command, so the call fails).

## 2026-11-24: released 0.4.2

What changed: Released with `scripts/weirdnet_release.py` (approved by Michael, 2026-11-24). Release commit [`ae57b61`](https://github.com/WeirdBrains/weirdnet-backends/commit/ae57b61), tag [`v0.4.2`](https://github.com/WeirdBrains/weirdnet-backends/releases/tag/v0.4.2), on [crates.io](https://crates.io/crates/weirdnet-backends/0.4.2) and [docs.rs](https://docs.rs/weirdnet-backends/0.4.2). See the entry below and [CHANGELOG.md](CHANGELOG.md).

## 2026-11-24: engine port race (0.4.2)

What changed (see [CHANGELOG.md](CHANGELOG.md)): the supervisor tells each engine a port it picked and freed, so another process could bind it first (seen in Ymir's tests under load). Now readiness also checks identity: after `/health`, `GET /v1/models` with the launch's key must answer 2xx and list the launch id (alias, `aliases` or a module); 401/403 or another model list means a stranger holds the port. That try's engine is stopped and the launch goes again on a new port (`PORT_TRIES` = 3). An engine that exits while loading is retried too when its port is held by someone else now or its log since the launch says the address was in use ("couldn't bind", "address already in use", errno 98/48, ...); such a try isn't a crash (no `Crashed` event, no crash count). Engine-agnostic, so no parsing of `--port 0` output (llama-server b12628 can bind port 0 and logs it, vLLM can't report it cleanly). MLX-LM: the advertised port is the proxy's, bound before the engine starts; the engine's private port gets the exit-and-retry path only (MLX-LM has no key to check). Wrapped servers: nothing picked, unchanged. Bug fix, so a patch release.

Checks run: fmt; clippy `--all-features --all-targets -D warnings`; tests `--all-features` (33 unit, 11 adapter, 2 node). New: `a_taken_port_is_retried_on_another` (mock modes: a stranger answering with another key, for llama.cpp and vLLM; "couldn't bind" in the log and exit; another process holding the port), unit tests for the model-list and log checks. The mock engine serves `/v1/models` behind its key.

Not tested: real llama-server and vLLM racing for a port. Checked against llama.cpp b12628 source: `/v1/models` is behind the API key (only `/health` is public) and lists the `--alias` as `id`; vLLM puts `/v1/*` behind `VLLM_API_KEY` and lists `--served-model-name`.

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
