# weirdnet-backends agents

The collective `AGENTS.md` in `WeirdBrains/weirdnet-agents` is mandatory and wins on conflict.

- No app-specific types or names. The node decides nothing beyond itself: what to load is the
  caller's decision; cross-machine group workers are the application's `GroupWorkers`.
- Engines bind loopback only, with a per-process API key in the environment, never argv.
- Unix-only process handling stays behind `cfg(unix)`; the library must build on Windows.
- Tests use the self-hosted mock engine (`tests/common/mock.rs`), never a real engine.
- Build outside the source checkout, with `CARGO_TARGET_DIR` on a scratch disk; clean targets afterwards.
