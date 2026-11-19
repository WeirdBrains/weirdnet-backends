# weirdnet-backends

The Weirdnet model-node wrapper. A model node runs models and takes requests, like a llama.cpp
or vLLM server (it can even be one: a stock server it only wraps). This crate starts and stops
the engines and serves them to inference nodes; what to load is always the caller's decision.

- `EngineConfig` / `ModelLaunch`: how `llama-server` and `vllm serve` are launched (loopback
  only, random port, per-process API key in the environment) and probed. `Wrap` attaches to a
  stock OpenAI-compatible server on loopback without owning it.
- `Supervisor`: load and unload with draining, **swap groups** (loading one model of a group
  unloads the others), **idle unload**, crash restarts with exponential backoff and crash-loop
  detection, `SupervisorEvent`s, per-model logs; chat, completion and embedding forwarding.
- `ChunkDecoder` / `ToolCalls`: OpenAI server-sent events decoded into `Chunk`s (text,
  reasoning, tool-call fragments, finish reason and usage).
- `ModelHost`: a [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent) 0.4 `InferHost`
  serving **raw infer/2** from the supervisor's models. Cancel aborts the engine request;
  assistant prefixes continue (resume); draining refuses new requests.

```rust
use std::sync::Arc;
use weirdnet_backends::{EngineConfig, EngineKind, ModelHost, ModelLaunch, Supervisor, SupervisorOptions};

let chat = ModelLaunch::new("qwen3-8b", EngineKind::Llamacpp, "/models/qwen3-8b-q4_k_m.gguf");
let sup = Supervisor::new(vec![EngineConfig::llamacpp("llama-server")], vec![chat], SupervisorOptions::default());
sup.load("qwen3-8b").await?;
let host = Arc::new(ModelHost::new(sup.clone(), auth)); // serve with weirdnet_agent::remote::infer::v2::serve
```

## The `node` feature: `/weirdnet-backends/node/1`

How inference nodes drive model nodes over [`weirdnet-net`](https://crates.io/crates/weirdnet-net).

- **Capability records**, pushed to every subscriber: hardware and memory budget
  ([`weirdnet-model`](https://crates.io/crates/weirdnet-model) `hw`), engines, every
  configured model (state, revision, quantization, context and slot limits, load-time
  estimate, measured prefill/decode speed and time to first token), load (queued, active,
  slots, in flight), warm prefix hashes, swap groups, draining and whether the node takes
  group jobs. Each is signed by the node key, chained to the user by its `NodeCert`, carries
  an increasing sequence number and expires after 90 s; it is refreshed every 15 s and on
  every change. `Verifier` checks the signature, the signer, the expiry and the user's
  certificate; `Subscription` also rejects sequence numbers that go backwards.
- **Commands**: `Load` (with `force` for a crash loop), `Unload`, `Drain` (stop admissions and
  wait for requests under way, or resume), `GroupUp`/`GroupDown`, `Health`. Cross-machine swap
  groups (a Ray worker, an `rpc-server` on a private link) are started by the application's
  `GroupWorkers`; without one a node answers `Unsupported` and advertises `group_jobs: false`.
- `NodeServer::start` serves node/1 and raw `/weirdnet-agent/infer/2` on a `weirdnet-net` node
  and announces `weirdnet-backends-node`; `NodeService::withdraw` ends subscriptions when the
  node leaves. `NodeClient` is the inference node's side, over any `weirdnet-agent` `Dialer`.
- Who may drive a model node is an `Authorize` hook; `weirdnet_agent::serve::RosterAuth`
  admits the user's linked devices.

## Platforms

| | Linux | macOS | Windows |
|---|---|---|---|
| llama.cpp | yes | yes | yes |
| vLLM | yes | no (vLLM is Linux-only) | no |
| Graceful stop (SIGTERM, then SIGKILL after a grace period) | yes | yes | no: killed outright |
| Engine in its own process group (children stopped too) | yes | yes | no |
| Engine dies with the node process | yes (`PR_SET_PDEATHSIG`) | no | no |
| `node` feature | yes | yes | yes |

The Unix-only pieces are behind `cfg(unix)`; the library builds everywhere.

## Tests

The adapter tests run against a mock engine (the test binary re-executes itself as a fake
`llama-server`/`vllm serve`), never a real one. `tests/node.rs` runs node/1 and raw infer/2
between in-process `weirdnet-net` nodes on the memory transport.
