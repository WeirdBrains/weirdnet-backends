# Changelog

## [Unreleased]

### Added

- Engine adapters for llama.cpp `llama-server` and `vllm serve` (`EngineConfig`, `ModelLaunch`), and `Wrap` for a stock OpenAI-compatible server on loopback.
- `Supervisor`: loads on a random loopback port with a per-process API key and a readiness probe; drain and unload; swap groups; idle unload (`reap_idle`, `spawn_reaper`); crash restarts with backoff and crash loops; `SupervisorEvent`s and per-model logs; chat, completion and embedding forwarding.
- SSE decoding into `Chunk`s (text, reasoning, tool-call fragments, finish and usage) and `ToolCalls` to join streamed calls.
- OpenAI request bodies from `weirdnet-agent` 0.4 infer/2 requests, including assistant-prefix continuation.
- `ModelHost`: a `weirdnet-agent` `InferHost` serving raw infer/2 (and embeddings) from the supervisor's models, with draining and measured speed.
- `node` feature: `/weirdnet-backends/node/1` with signed capability records (node key, user certificate, sequence, 90 s expiry, 15 s refresh), `Load`, `Unload`, `Drain`, `GroupUp`/`GroupDown` through the application's `GroupWorkers`, health, `NodeServer` (node/1 plus raw infer/2 on a `weirdnet-net` node) and `NodeClient`.
