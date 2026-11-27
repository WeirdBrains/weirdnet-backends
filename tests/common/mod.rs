//! Shared test support: the mock engine (built from `examples/mock_engine.rs` by `cargo test`).
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use weirdnet_agent::remote::infer::v2;
use weirdnet_agent::remote::infer::{Admission, InferHost};
use weirdnet_agent::remote::{Caller, MemoryDialer};
use weirdnet_backends::{EngineConfig, EngineKind, ModelLaunch, Supervisor, SupervisorOptions};

pub mod mock;

/// The fake engine: this test binary, which turns into [`mock`] when started with `--port`.
pub fn mock_engine() -> PathBuf {
    std::env::current_exe().unwrap()
}

/// The test binaries' `main` (`harness = false`): the mock engine when started with engine
/// flags, else each test in turn (an argument filters by name).
pub fn main(tests: &[(&str, fn())]) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--port") {
        mock::main();
        return;
    }
    let filter = args.iter().find(|a| !a.starts_with('-'));
    let mut ran = 0;
    for (name, test) in tests {
        if filter.is_some_and(|f| !name.contains(f.as_str())) {
            continue;
        }
        println!("test {name} ...");
        test();
        println!("  ok");
        ran += 1;
    }
    println!("test result: ok. {ran} passed");
}

/// Runs `f` on a fresh multi-thread runtime.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap().block_on(f)
}

pub fn opts() -> SupervisorOptions {
    SupervisorOptions {
        stop_grace: Duration::from_secs(2),
        drain_timeout: Duration::from_secs(2),
        backoff_base: Duration::from_millis(100),
        backoff_max: Duration::from_millis(400),
        max_crashes: 2,
        ..Default::default()
    }
}

pub fn launch(id: &str, engine: EngineKind) -> ModelLaunch {
    ModelLaunch::new(id, engine, format!("/models/{id}.gguf"))
}

pub fn supervisor(models: Vec<ModelLaunch>) -> Supervisor {
    let mock = mock_engine();
    let engines = vec![EngineConfig::llamacpp(&mock), EngineConfig::vllm(&mock), EngineConfig::mlx(&mock), EngineConfig::vllm_omni(&mock)];
    Supervisor::new(engines, models, opts())
}

/// A memory dialer with `node` serving infer/2 from `host`.
pub fn serve(node: &str, host: Arc<dyn InferHost>, admission: Admission) -> MemoryDialer {
    let dialer = MemoryDialer::new();
    dialer.route_only(node, &[v2::PROTOCOL], move |ch, _proto| {
        let (h, a) = (host.clone(), admission.clone());
        tokio::spawn(async move {
            let _ = v2::serve(ch, Caller::new("inference-node"), h.as_ref(), &a).await;
        });
    });
    dialer
}
