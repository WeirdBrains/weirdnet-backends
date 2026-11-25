# Changelog

## [Unreleased]

### Added

- `ModelEntry::origin` and `license` in capability records, set per model with `NodeService::set_model_origin` (`hub`, `upload` or `train`, as in a weirdnet-model provenance record, and an SPDX license). Left out of the record when unset.

## [0.4.4] - 2026-11-24

### Added

- models/1 uploads: `ModelsCommand::Upload` with `Upload` and `UploadFrame` (48 KiB `CHUNK`s), `ModelsReply::UploadReady` and `Uploaded`, and `ModelsClient::upload`. The node checks names, size, free space, engine and base up front, headers as the first bytes arrive and every SHA-256 at the end, then places the files with weirdnet-model 0.5.3 `ModelStore::import` and can serve them: GGUF, safetensors and MLX models, adapters (live on their base) and vision projectors. Closed uploads resume from staged parts (`with_staging`, swept after `WritePolicy::part_ttl`); `ModelsCommand::AbandonUpload` drops them.
- Copies between nodes: `ModelsCommand::Fetch` with `FetchReady`, `Chunk` and `FetchDone`, `PullSource::Peer` and `Pull::from_peer` (`ModelsService::with_peer_dialer`), and `ModelsClient::replicate`.
- `WritePolicy` (a separate write `Authorize`, `max_upload`, `quota`, `KeepFree`) for uploads, removes and peer copies; by default any caller the service admits may write. Stable refusal reasons in `refusal` with `refusal_reason`.
- `ModelsCommand::Disk` and `DiskReport`, `ModelsClient::disk`.
- `StoredEntry::projector` and `provenance()`; `List` includes safetensors and MLX models and projectors. `Pulled::revision` and `projector_for`; `Pull::source` and `projector_for`.
- `ModelLaunch::mmproj` (`with_mmproj`): llama.cpp starts with `--mmproj`, and `capabilities()` reports image input. `Supervisor::set_projector_live`.

### Changed

- Needs weirdnet-model 0.5.3.
- `Remove` follows the `WritePolicy`. A served model is reloaded when its projector changes. Nodes before 0.4.4 refuse the new commands.

## [0.4.3] - 2026-11-24

### Added

- Node reservations: `Supervisor::reserve(holder)` holds the node's GPU for one job. It stops every engine and starts none (loads, requests, crash restarts) until `Supervisor::release(holder)`. `Reservation`, `Supervisor::reservation` and `reservation_changes`.
- node/1 `Command::Reserve` and `Command::Release`, `NodeClient::reserve` and `release`, and `Capability::reserved` and `Health::reserved`, so placement can skip a reserved node. A reserved node refuses `GroupUp`. Nodes before 0.4.3 fail the new commands.

## [0.4.2] - 2026-11-24

### Fixed

- A launched engine could lose the port it was told to another process (the supervisor picks a free port, then the engine binds it later). The supervisor now checks that the server answering on the port is its engine before calling it ready: `/v1/models` must take the launch's API key and list its id (llama-server and vLLM). If another server answers there, or the engine exits while loading because the port was taken (something else holds it, or its log says the address is in use), the launch is tried again on a new port, up to 3 times, and the failed try is not counted as a crash. MLX-LM's advertised port is the proxy's, bound before the engine starts, so only its engine's private port can be retried that way. Wrapped servers are unchanged.

## [0.4.1] - 2026-11-23

### Added

- Live adapter changes: `Supervisor::add_adapter_live` and `remove_adapter_live` load and unload an adapter in a running vLLM engine (`/v1/load_lora_adapter`, `/v1/unload_lora_adapter`) without restarting the base. vLLM with LoRA on now starts with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; `ModelLaunch::lora_slots` (`with_lora_slots`, `runtime_lora`) turns LoRA on without adapters at start.
- `Supervisor::set_adapter_scale` changes a llama.cpp adapter's scale from the next request, with no restart. `Supervisor::chat` takes weirdnet-agent 0.5.2's `Request::adapter_scale` for one request, and `complete_scaled` does the same for completions; raw infer/2 passes it through. vLLM refuses both. `MAX_ADAPTER_SCALE`.
- `models` (feature `node`): `ModelsCommand::ServeAdapter`, `UnserveAdapter` and `AdapterScale` with their replies, `ModelsService::serve_adapter`, `unserve_adapter` and `set_adapter_scale`, and `ModelsClient::serve_adapter`, `unserve_adapter` and `set_adapter_scale`. Pulling an adapter for a base and removing one now use the live calls. Nodes before 0.4.1 refuse the new commands.

### Changed

- llama.cpp: an adapter added to a running base whose engine already has its file (removed and added back, under any id) is `AdapterChange::Live`, not `NextLoad`. Requests find an adapter in the engine by file on llama.cpp.
- Needs weirdnet-agent 0.5.2.

## [0.4.0] - 2026-11-22

### Added

- LoRA adapters: `LoraAdapter` (`new`, `with_scale`), `ModelLaunch::adapters`, `ModelLaunch::with_adapters`, `served_ids` and `check_adapters`. llama.cpp loads GGUF LoRA files with `--lora` and `--lora-init-without-apply` and each request sets its adapter in `lora`; vLLM gets `--enable-lora`, one `--lora-modules` list and `--max-loras`.
- `openai::Target` (`Base`, `Adapter`) with `chat_body_for` and `completion_body_for`.
- `Supervisor::add_adapter` (`AdapterChange::Live` or `NextLoad`) and `Supervisor::remove_adapter`. Chat, completion, state and capabilities resolve an adapter to its base; `load(adapter)` loads the base; `unload(adapter)` and `embed(adapter)` are refused; `remove_model(base)` removes its adapters. `ModelStatus::base` and a status row per adapter.
- `models` (feature `node`): `Pull::adapter_for` (`with_adapter_for`) pulls an adapter for a served base and, with `serve`, adds it; `RepoFits::adapters`; `List` and `Remove` cover adapters. Adapters whose architecture or base model does not match are refused.

### Changed

- Breaking: `ModelLaunch` is `#[non_exhaustive]`; build it with `ModelLaunch::new` and the `with_` methods.
- Breaking: `models::StoredEntry::model` is an `Option` and `StoredEntry` gains `adapter` and `base`; `StoredEntry` and `RepoFits` no longer implement `Eq`. 0.3 clients cannot decode a `List` reply that has adapters.
- `weirdnet-model` (0.5.2) is now always a dependency, not only with `node`, to read GGUF adapter headers.

## [0.3.0] - 2026-11-21

### Added

- `models` (feature `node`): `/weirdnet-backends/models/1` for the owner only (`ModelsService` takes an `Authorize`, such as `RosterAuth`): Hub search, a repo's files measured against the node's budget, pull with progress (split GGUFs too, the largest that fits by default) and serve without a restart, list and remove. `ModelsServer` and `ModelsClient`. The node's Hugging Face token stays on the node.
- `Supervisor::add_model` and `Supervisor::remove_model`: serve a new model, or stop serving one, without a restart (`SupervisorEvent::Added` and `Removed`); `Supervisor::launches`.

### Changed

- Breaking: the `node` feature needs `weirdnet-model` 0.5.1 and no longer takes 0.4. `hw::Hardware` and `hw::Budget` in `Capability` and `NodeService::with_hardware` are its types, so apps on weirdnet-model 0.4 move to 0.5.1 with it.

## [0.2.0] - 2026-11-20

### Added

- Image and audio content parts (`weirdnet-agent` 0.5 `Content`) reach llama.cpp unchanged; `ModelHost` refuses them for a model without those capabilities (`Unsupported`) and refuses remote image URLs (`BadRequest`).
- `Capabilities::inputs` and `Capabilities::lacking`.

### Changed

- Breaking: needs `weirdnet-agent` 0.5 (content parts).

## [0.1.3] - 2026-11-20

### Added

- MLX-LM engine (`EngineKind::Mlx`, `EngineConfig::mlx`) for Apple silicon: `mlx_lm.server` behind a proxy that checks the node's API key, rewrites `model` to `default_model`, refuses image, audio and embedding requests, and reports ready only after a one-token completion. Wrapped MLX-LM servers get the same proxy.
- `mlx` module: `install` puts a pinned `mlx-lm` (0.32.2) into a venv with `uv` or `python3`; `install_commands`, `server_in`, `supported`.
- `Capabilities` (chat, embeddings, vision, audio): `EngineKind::capabilities`, `ModelLaunch::capabilities`, `Supervisor::capabilities` and `ModelStatus::capabilities` in capability records.

### Changed

- `Supervisor::embed` refuses MLX models with a bad request.

## [0.1.2] - 2026-11-20

### Added

- Link info in capability records: `Capability::links` (`LinkInfo`, `LinkMeasure`, `AddrClass`), set with `NodeService::set_links`; `link::local_links` lists this machine's private interfaces with the NIC's nominal speed. Only loopback, link-local and private addresses are advertised. Old verifiers ignore the field.
- `Verifier::with_roster`: also refuses records from nodes that are unknown to, or revoked in, the `weirdnet-sync` roster.

### Changed

- Needs weirdnet-agent 0.4.4 and accepts weirdnet-sync 0.5 up to 0.7 (`>=0.5, <0.8`), so apps on sync 0.6 or 0.7 share one roster type.

## [0.1.1] - 2026-11-19

### Added

- `ModelHost` serves raw prompts (`weirdnet-agent` 0.4.1 `Input::Prompt`) through the engine's `/v1/completions`; a resume continues the prompt plus the assistant prefix. Response formats and tools don't apply to raw prompts.
- `ModelHost` embeddings report the engine's prompt tokens (`EmbedBackend::embed_counted`).
- `RAW_INFER_PROTOCOL` (`/weirdnet-backends/infer/1`): raw inference in the infer/2 framing under its own id, so one node can be an inference node and a model node. `NodeServer` always serves it and lists it in the model-node announcement.
- `NodeServerOptions` and `NodeServer::start_with`: `legacy_infer2` (default on) keeps serving raw inference on `/weirdnet-agent/infer/2` for 0.1.0 inference nodes; turn it off on a node that is also an inference node.
- `RawInferDialer`: wraps the dialer of an inference node's model-node `Pool`; infer/2 opens try `/weirdnet-backends/infer/1` first and fall back to infer/2 on nodes that don't accept it (remembered per node).

### Changed

- Depends on `weirdnet-agent` `"0.4.1"` and `weirdnet-model` `"0.4"` (caret requirements, not exact pins).

## [0.1.0] - 2026-11-19

### Added

- Engine adapters for llama.cpp `llama-server` and `vllm serve` (`EngineConfig`, `ModelLaunch`), and `Wrap` for a stock OpenAI-compatible server on loopback.
- `Supervisor`: loads on a random loopback port with a per-process API key and a readiness probe; drain and unload; swap groups; idle unload (`reap_idle`, `spawn_reaper`); crash restarts with backoff and crash loops; `SupervisorEvent`s and per-model logs; chat, completion and embedding forwarding.
- SSE decoding into `Chunk`s (text, reasoning, tool-call fragments, finish and usage) and `ToolCalls` to join streamed calls.
- OpenAI request bodies from `weirdnet-agent` 0.4 infer/2 requests, including assistant-prefix continuation.
- `ModelHost`: a `weirdnet-agent` `InferHost` serving raw infer/2 (and embeddings) from the supervisor's models, with draining and measured speed.
- `node` feature: `/weirdnet-backends/node/1` with signed capability records (node key, user certificate, sequence, 90 s expiry, 15 s refresh), `Load`, `Unload`, `Drain`, `GroupUp`/`GroupDown` through the application's `GroupWorkers`, health, `NodeServer` (node/1 plus raw infer/2 on a `weirdnet-net` node) and `NodeClient`.
