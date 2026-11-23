//! The adapters against a fake engine (`examples/mock_engine.rs`): launch, readiness, streamed
//! chat with reasoning, tool calls and prefixes, completions, embeddings, swap groups, idle
//! unload, wrapping a stock server, crash restarts and MLX-LM behind its proxy; and raw infer/2
//! through `ModelHost`.

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use weirdnet_agent::remote::infer::Admission;
use weirdnet_agent::remote::infer::v2::{EmbedRequest, Prefix, Request, Sampling};
use weirdnet_agent::remote::pool::{NoEvents, Pool};
use weirdnet_agent::remote::{Authorize, Caller, RemoteError};
use weirdnet_agent::{CancelToken, ContentPart, Message, ToolSpec};
use weirdnet_backends::{
    AdapterChange, Capabilities, Chunk, EngineKind, LoraAdapter, ModelHost, ModelLaunch, ModelState, RAW_INFER_PROTOCOL, RawInferDialer, Supervisor,
    SupervisorEvent, Wrap,
};

use common::{launch, serve, supervisor};

async fn collect(sup: &Supervisor, id: &str, req: &Request) -> Vec<Chunk> {
    let mut s = sup.chat(id, req).await.unwrap();
    let mut out = Vec::new();
    while let Some(c) = s.next().await {
        out.push(c.unwrap());
    }
    out
}

fn text(chunks: &[Chunk]) -> String {
    chunks.iter().filter_map(|c| if let Chunk::Text(t) = c { Some(t.as_str()) } else { None }).collect()
}

fn chat_reasoning_tools_prefix_completion_and_embeddings() {
    common::block_on(async {
        let mut embed = launch("embed", EngineKind::Llamacpp);
        embed.embedding = true;
        let sup = supervisor(vec![launch("chat", EngineKind::Llamacpp), launch("big", EngineKind::Vllm), embed]);
        assert_eq!(sup.state("chat"), Some(ModelState::Unloaded));
        let err = sup.chat("chat", &Request::messages(vec![Message::user("hi")])).await.err().unwrap();
        assert!(matches!(err, RemoteError::Unavailable { .. }), "{err:?}");

        sup.load("chat").await.unwrap();
        assert_eq!(sup.state("chat"), Some(ModelState::Ready));

        let chunks = collect(&sup, "chat", &Request::messages(vec![Message::user("think about it")])).await;
        assert_eq!(chunks.first(), Some(&Chunk::Reasoning("hmm".into())));
        assert_eq!(text(&chunks), "Hello from chat! You said: think about it");
        match chunks.last().unwrap() {
            Chunk::Done { finish_reason, usage } => {
                assert_eq!(finish_reason.as_deref(), Some("stop"));
                assert_eq!(usage.unwrap().prompt_tokens, 5);
            }
            other => panic!("{other:?}"),
        }

        let req = Request::messages(vec![Message::user("x")]).with_tools(vec![ToolSpec::new("lookup", "look it up")]);
        let mut calls = weirdnet_backends::ToolCalls::default();
        for c in collect(&sup, "chat", &req).await {
            if let Chunk::ToolCall { index, id, name, arguments } = c {
                calls.push(index, id, name, &arguments);
            }
        }
        let calls = calls.finish();
        assert_eq!((calls[0].id.as_str(), calls[0].name.as_str()), ("call_1", "lookup"));
        assert_eq!(calls[0].arguments, serde_json::json!({"q": "x"}));

        let req = Request::messages(vec![Message::user("hi")]).with_prefix(Prefix::new("Hello ", ""));
        assert_eq!(text(&collect(&sup, "chat", &req).await), "world");

        let mut s = sup.complete("chat", "Tell me", &Sampling::default(), None).await.unwrap();
        let mut done = String::new();
        while let Some(c) = s.next().await {
            if let Chunk::Text(t) = c.unwrap() {
                done.push_str(&t);
            }
        }
        assert_eq!(done, " once upon a time");

        // vLLM's flags and key variable.
        sup.load("big").await.unwrap();
        assert_eq!(text(&collect(&sup, "big", &Request::messages(vec![Message::user("yo")])).await), "Hello from big! You said: yo");

        sup.load("embed").await.unwrap();
        let (vectors, tokens) = sup.embed("embed", &["one two".into(), "three".into()]).await.unwrap();
        assert_eq!((vectors, tokens), (vec![vec![7.0, 2.0, 1.0, 0.0], vec![5.0, 1.0, 1.0, 0.0]], 2));
        sup.shutdown().await;
        assert_eq!(sup.state("chat"), Some(ModelState::Unloaded));
    })
}

fn swap_groups_and_idle_unload() {
    common::block_on(async {
        let mut a = launch("a", EngineKind::Llamacpp);
        a.swap_group = Some("gpu".into());
        let mut b = launch("b", EngineKind::Llamacpp);
        b.swap_group = Some("gpu".into());
        let mut idle = launch("idle", EngineKind::Llamacpp);
        idle.idle_ttl = Some(Duration::from_millis(200));
        let sup = supervisor(vec![a, b, idle]);
        let mut events = sup.subscribe();
        sup.load("a").await.unwrap();
        sup.load("b").await.unwrap();
        assert_eq!((sup.state("a"), sup.state("b")), (Some(ModelState::Unloaded), Some(ModelState::Ready)), "one model per swap group");

        sup.load("idle").await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        assert_eq!(sup.reap_idle().await, vec!["idle".to_string()]);
        assert_eq!(sup.state("idle"), Some(ModelState::Unloaded));
        let mut seen = Vec::new();
        while let Ok(e) = events.try_recv() {
            seen.push(e);
        }
        assert!(seen.contains(&SupervisorEvent::Unloaded { model: "idle".into(), idle: true }), "{seen:?}");
        sup.shutdown().await;
    })
}

fn wraps_a_loopback_server_and_refuses_others() {
    common::block_on(async {
        // Run the mock as a "stock server" the node does not own.
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let mut child = tokio::process::Command::new(common::mock_engine())
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--alias", "stock"])
            .env("WEIRDNET_MOCK_LOAD_MS", "0")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut stock = launch("stock", EngineKind::Llamacpp);
        stock.wrap = Some(Wrap { url: format!("http://127.0.0.1:{port}"), api_key: None });
        let mut remote = launch("remote", EngineKind::Llamacpp);
        remote.wrap = Some(Wrap { url: "http://192.0.2.1:8080".into(), api_key: None });
        let sup = supervisor(vec![stock, remote]);
        let mut ok = false;
        for _ in 0..50 {
            if sup.load("stock").await.is_ok() {
                ok = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(ok, "wrapped server never came up");
        assert!(sup.status().iter().any(|m| m.id == "stock" && m.wrapped && m.pid.is_none()));
        assert_eq!(text(&collect(&sup, "stock", &Request::messages(vec![Message::user("hi")])).await), "Hello from stock! You said: hi");
        assert!(sup.load("remote").await.is_err(), "non-loopback wrap is refused");
        sup.unload("stock").await.unwrap();
        assert!(child.try_wait().unwrap().is_none(), "unload only detaches a wrapped server");
        child.kill().await.unwrap();
    })
}

#[cfg(unix)]
fn crash_restarts_then_crash_loop() {
    common::block_on(async {
        let sup = supervisor(vec![launch("crashy", EngineKind::Llamacpp)]);
        let mut events = sup.subscribe();
        let kill = |sup: &Supervisor| {
            let pid = sup.status()[0].pid.unwrap();
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        };
        sup.load("crashy").await.unwrap();
        kill(&sup);
        // Restarted after the backoff.
        let mut restarted = false;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            if sup.state("crashy") == Some(ModelState::Ready) && sup.status()[0].crashes == 1 {
                restarted = true;
                break;
            }
        }
        assert!(restarted, "{:?}", sup.status());
        kill(&sup);
        let mut looped = false;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(100)).await;
            while let Ok(e) = events.try_recv() {
                looped |= matches!(e, SupervisorEvent::CrashLoop { .. });
            }
            if looped {
                break;
            }
        }
        assert!(looped);
        assert!(sup.load("crashy").await.is_err(), "a crash loop refuses plain loads");
        sup.load_with("crashy", true).await.unwrap();
        sup.shutdown().await;
    })
}

fn raw_infer2_through_the_model_host() {
    common::block_on(async {
        let mut embed = launch("embed", EngineKind::Llamacpp);
        embed.embedding = true;
        let sup = supervisor(vec![launch("chat", EngineKind::Llamacpp), launch("cold", EngineKind::Llamacpp), embed]);
        sup.load("chat").await.unwrap();
        sup.load("embed").await.unwrap();
        let auth: Arc<dyn Authorize> =
            Arc::new(|c: &Caller| if c.peer == "inference-node" { Ok(()) } else { Err(RemoteError::refused("unknown-device")) });
        let host = Arc::new(ModelHost::new(sup.clone(), auth).with_default("chat"));
        let dialer = serve("model-node", host.clone(), Admission::new(2));
        let pool = Pool::new(Arc::new(dialer), vec!["model-node".to_string()]);
        let cancel = CancelToken::new();

        let answer = pool.infer(Request::messages(vec![Message::user("think hi")]).raw(), &NoEvents, &cancel).await.unwrap();
        assert_eq!((answer.text.as_str(), answer.reasoning.as_str()), ("Hello from chat! You said: think hi", "hmm"));
        assert_eq!(answer.finish_reason.as_deref(), Some("stop"));
        assert_eq!(answer.usage.unwrap().completion_tokens, 8);
        assert!(host.speed("chat").ttft_ms.is_some());

        let tools = Request::messages(vec![Message::user("q")]).with_tools(vec![ToolSpec::new("lookup", "")]).raw();
        let answer = pool.infer(tools, &NoEvents, &cancel).await.unwrap();
        assert_eq!((answer.tool_calls.len(), answer.finish_reason.as_deref()), (1, Some("tool_calls")));

        // Not loaded: refused, unless the request waits for the model.
        let cold = Request::messages(vec![Message::user("hi")]).with_model("cold").raw();
        let err = pool.infer(cold.clone(), &NoEvents, &cancel).await.unwrap_err();
        assert!(matches!(err.error, RemoteError::Unavailable { .. }), "{:?}", err.error);
        let mut waiting = cold;
        waiting.wait_for_model = true;
        assert_eq!(pool.infer(waiting, &NoEvents, &cancel).await.unwrap().text, "Hello from cold! You said: hi");

        // A raw prompt goes to /v1/completions.
        let answer = pool.infer(Request::prompt("Long ago").raw(), &NoEvents, &cancel).await.unwrap();
        assert_eq!((answer.text.as_str(), answer.finish_reason.as_deref()), (" once upon a time", Some("length")));
        assert_eq!(answer.usage.map(|u| (u.prompt_tokens, u.completion_tokens)), Some((2, 4)));

        let e = pool.embed(EmbedRequest::new(vec!["a b".into(), "c".into()]).with_model("embed"), &cancel).await.unwrap();
        assert_eq!(e.vectors, vec![vec![3.0, 2.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 0.0]]);
        assert_eq!(e.usage.map(|u| u.prompt_tokens), Some(2), "one token per input from the mock engine");

        // RawInferDialer: this memory node serves only infer/2 (a 0.1.0 model node), so the
        // dialer falls back and remembers it; a node on the new id is used directly.
        let raw = Arc::new(RawInferDialer::new(Arc::new(serve("model-node", host.clone(), Admission::new(2)))));
        let p = Pool::new(raw.clone(), vec!["model-node".to_string()]);
        assert_eq!(
            p.infer(Request::messages(vec![Message::user("hi")]).raw(), &NoEvents, &cancel).await.unwrap().text,
            "Hello from chat! You said: hi"
        );
        assert!(raw.is_legacy("model-node"));
        let new_only = weirdnet_agent::remote::MemoryDialer::new();
        let h = host.clone();
        new_only.route_only("new", &[RAW_INFER_PROTOCOL], move |ch, _| {
            let h = h.clone();
            tokio::spawn(async move {
                let _ = weirdnet_agent::remote::infer::v2::serve(ch, Caller::new("inference-node"), h.as_ref(), &Admission::new(1)).await;
            });
        });
        let raw = Arc::new(RawInferDialer::new(Arc::new(new_only)));
        let p = Pool::new(raw.clone(), vec!["new".to_string()]);
        assert_eq!(
            p.infer(Request::messages(vec![Message::user("yo")]).raw(), &NoEvents, &cancel).await.unwrap().text,
            "Hello from chat! You said: yo"
        );
        assert!(!raw.is_legacy("new"));

        host.set_draining(true);
        let err = pool.infer(Request::messages(vec![Message::user("hi")]).raw(), &NoEvents, &cancel).await.unwrap_err();
        assert!(matches!(err.error, RemoteError::Unavailable { .. }), "{:?}", err.error);
        sup.shutdown().await;
    })
}

fn mlx_behind_its_proxy() {
    common::block_on(async {
        let mut emb = launch("mlx-embed", EngineKind::Mlx);
        emb.embedding = true;
        let sup = supervisor(vec![launch("qwen", EngineKind::Mlx), emb, launch("gguf", EngineKind::Llamacpp)]);
        // The fake MLX-LM answers /health at once but completions only after this long; the
        // supervisor must wait for a real token.
        let started = std::time::Instant::now();
        sup.load("qwen").await.unwrap();
        assert!(started.elapsed() >= Duration::from_millis(200), "ready before the weights were in");
        assert_eq!(sup.state("qwen"), Some(ModelState::Ready));
        // The request names the pool id; the engine only knows `default_model`.
        let chunks = collect(&sup, "qwen", &Request::messages(vec![Message::user("hi")])).await;
        assert_eq!(text(&chunks), "Hello from default_model! You said: hi");
        let mut s = sup.complete("qwen", "Tell me", &Sampling::default(), None).await.unwrap();
        let mut out = String::new();
        while let Some(c) = s.next().await {
            if let Chunk::Text(t) = c.unwrap() {
                out.push_str(&t);
            }
        }
        assert_eq!(out, " once upon a time");

        let err = sup.embed("qwen", &["x".into()]).await.unwrap_err();
        assert!(matches!(err, RemoteError::BadRequest { .. }), "{err:?}");
        let err = sup.load("mlx-embed").await.unwrap_err();
        assert!(matches!(err, RemoteError::BadRequest { .. }), "{err:?}");
        assert!(matches!(sup.state("mlx-embed"), Some(ModelState::Failed(_))));

        let caps = |id: &str| sup.status().into_iter().find(|m| m.id == id).unwrap().capabilities.unwrap();
        assert_eq!(caps("qwen"), Capabilities::text());
        assert!(!caps("qwen").vision && !caps("qwen").audio && !caps("qwen").embeddings);
        assert_eq!(sup.capabilities("gguf"), Some(Capabilities::text()));

        sup.unload("qwen").await.unwrap();
        assert_eq!(sup.state("qwen"), Some(ModelState::Unloaded));
        sup.shutdown().await;

        // A stock mlx_lm.server the node only wraps still goes through the proxy.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut child = tokio::process::Command::new(common::mock_engine())
            .args(["--model", "/m/x", "--host", "127.0.0.1", "--port", &port.to_string(), "--max-tokens", "64"])
            .env("WEIRDNET_MOCK_LOAD_MS", "0")
            .env_remove("LLAMA_API_KEY")
            .env_remove("VLLM_API_KEY")
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let mut stock = launch("stock-mlx", EngineKind::Mlx);
        stock.wrap = Some(Wrap { url: format!("http://127.0.0.1:{port}"), api_key: None });
        let sup = supervisor(vec![stock]);
        let mut ok = false;
        for _ in 0..50 {
            if sup.load("stock-mlx").await.is_ok() {
                ok = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(ok, "wrapped MLX server never came up");
        let chunks = collect(&sup, "stock-mlx", &Request::messages(vec![Message::user("yo")])).await;
        assert_eq!(text(&chunks), "Hello from default_model! You said: yo");
        sup.shutdown().await;
        child.kill().await.unwrap();
    })
}

const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn ask(images: usize, audio: bool) -> Vec<Message> {
    let mut parts = vec![ContentPart::text("describe")];
    parts.extend((0..images).map(|_| ContentPart::image(PNG)));
    if audio {
        parts.push(ContentPart::audio("UklGRg==", "wav"));
    }
    vec![Message::user(parts)]
}

/// End to end: pool -> infer/2 -> `ModelHost` -> llama.cpp adapter -> a fake `llama-server`
/// started with `--mmproj`, which counts the images it receives.
fn images_and_audio_reach_llama_server_with_a_projector() {
    common::block_on(async {
        let mut vision = launch("vision", EngineKind::Llamacpp);
        vision.extra_args = vec!["--mmproj".into(), "/models/mmproj-vision.gguf".into()];
        let mut embed = launch("embed", EngineKind::Llamacpp);
        embed.extra_args = vision.extra_args.clone();
        embed.embedding = true;
        let sup = supervisor(vec![vision, launch("text", EngineKind::Llamacpp), embed]);
        sup.load("vision").await.unwrap();
        sup.load("text").await.unwrap();
        let status = sup.status();
        let caps = |id: &str| status.iter().find(|m| m.id == id).unwrap().capabilities.unwrap();
        assert_eq!(caps("vision"), Capabilities::text().with_vision(true).with_audio(true));
        assert_eq!((caps("text"), caps("embed")), (Capabilities::text(), Capabilities::embeddings()));

        let auth: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let host = Arc::new(ModelHost::new(sup.clone(), auth).with_default("text"));
        let pool = Pool::new(Arc::new(serve("model-node", host, Admission::new(2))), vec!["model-node".to_string()]);
        let cancel = CancelToken::new();
        for images in [1, 3] {
            let a = pool.infer(Request::messages(ask(images, false)).with_model("vision"), &NoEvents, &cancel).await.unwrap();
            assert_eq!(a.text, format!("Hello from vision! You said: describe [{images} image(s)]"));
        }
        let a = pool.infer(Request::messages(ask(1, true)).with_model("vision"), &NoEvents, &cancel).await.unwrap();
        assert_eq!(a.text, "Hello from vision! You said: describe [1 image(s)] [1 audio]");

        // Never on a text-only model, named or the default.
        for req in [Request::messages(ask(1, false)).with_model("text"), Request::messages(ask(1, false))] {
            let e = pool.infer(req, &NoEvents, &cancel).await.unwrap_err();
            assert!(matches!(&e.error, RemoteError::Unsupported { message } if message.contains("`text` does not take images")), "{:?}", e.error);
        }
        let e =
            pool.infer(Request::messages(vec![Message::user(vec![ContentPart::audio("UklGRg==", "wav")])]), &NoEvents, &cancel).await.unwrap_err();
        assert!(matches!(&e.error, RemoteError::Unsupported { message } if message.contains("does not take audio")), "{:?}", e.error);
        let remote = vec![Message::user(vec![ContentPart::image("https://example.com/cat.png")])];
        let e = pool.infer(Request::messages(remote).with_model("vision"), &NoEvents, &cancel).await.unwrap_err();
        assert!(matches!(&e.error, RemoteError::BadRequest { message } if message.contains("data:image")), "{:?}", e.error);
        // Text is unchanged.
        let a = pool.infer(Request::messages(vec![Message::user("hi")]), &NoEvents, &cancel).await.unwrap();
        assert_eq!(a.text, "Hello from text! You said: hi");
        sup.shutdown().await;
    })
}

fn models_added_and_removed_at_run_time() {
    common::block_on(async {
        let sup = supervisor(vec![launch("chat", EngineKind::Llamacpp)]);
        let mut events = sup.subscribe();
        assert!(sup.add_model(launch("chat", EngineKind::Llamacpp)).is_err(), "ids are unique");
        let mut bad = launch("bad", EngineKind::Llamacpp);
        bad.wrap = Some(Wrap { url: "http://example.com:8000".into(), api_key: None });
        assert!(sup.add_model(bad).is_err(), "wrapped servers stay on loopback");
        sup.add_model(launch("new", EngineKind::Llamacpp)).unwrap();
        assert_eq!(sup.state("new"), Some(ModelState::Unloaded));
        assert_eq!(sup.launches().iter().map(|l| l.id.as_str()).collect::<Vec<_>>(), ["chat", "new"]);
        sup.load("new").await.unwrap();
        assert_eq!(text(&collect(&sup, "new", &Request::messages(vec![Message::user("hi")])).await), "Hello from new! You said: hi");
        let gone = sup.remove_model("new").await.unwrap();
        assert_eq!(gone.id, "new");
        assert_eq!(sup.state("new"), None);
        assert!(sup.status().iter().all(|m| m.id != "new"));
        assert!(sup.remove_model("new").await.is_err());
        let mut seen = Vec::new();
        while let Ok(e) = events.try_recv() {
            seen.push(e);
        }
        assert!(seen.contains(&SupervisorEvent::Added { model: "new".into() }), "{seen:?}");
        assert_eq!(seen.last(), Some(&SupervisorEvent::Removed { model: "new".into() }), "{seen:?}");
        sup.shutdown().await;
    })
}

/// A GGUF header: `general.architecture` plus string and u32 keys.
fn gguf(arch: &str, kv: &[(&str, &str)], ints: &[(&str, u32)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(((1 + kv.len() + ints.len()) as u64).to_le_bytes());
    let s = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (k, v) in std::iter::once(&("general.architecture", arch)).chain(kv) {
        s(&mut out, k);
        out.extend(8u32.to_le_bytes());
        s(&mut out, v);
    }
    for (k, v) in ints {
        s(&mut out, k);
        out.extend(4u32.to_le_bytes());
        out.extend(v.to_le_bytes());
    }
    out
}

async fn reply(sup: &Supervisor, id: &str) -> Result<String, RemoteError> {
    let mut s = sup.chat(id, &Request::messages(vec![Message::user("hi")])).await?;
    let mut out = String::new();
    while let Some(c) = s.next().await {
        if let Chunk::Text(t) = c? {
            out.push_str(&t);
        }
    }
    Ok(out)
}

fn lora_adapters_on_a_base() {
    common::block_on(async {
        let tmp = tempfile::tempdir().unwrap();
        let w = |name: &str, b: Vec<u8>| {
            let p = tmp.path().join(name);
            std::fs::write(&p, b).unwrap();
            p.display().to_string()
        };
        let lora = || gguf("qwen3", &[("general.type", "adapter"), ("adapter.type", "lora")], &[]);
        let base = w("base.gguf", gguf("qwen3", &[], &[("qwen3.block_count", 1)]));
        let (a, b) = (w("a.gguf", lora()), w("b.gguf", lora()));
        let sup = supervisor(vec![]);
        let mut events = sup.subscribe();
        sup.add_model(ModelLaunch::new("qwen", EngineKind::Llamacpp, &base).with_adapters(vec![LoraAdapter::new("tune-a", &a).with_scale(0.5)]))
            .unwrap();
        assert!(matches!(events.recv().await.unwrap(), SupervisorEvent::Added { model } if model == "qwen"));
        assert!(matches!(events.recv().await.unwrap(), SupervisorEvent::Added { model } if model == "tune-a"));
        let rows = sup.status();
        assert_eq!(rows.iter().map(|r| (r.id.as_str(), r.base.as_deref())).collect::<Vec<_>>(), [("qwen", None), ("tune-a", Some("qwen"))]);
        assert_eq!(rows[1].capabilities, rows[0].capabilities);
        assert!(sup.add_model(ModelLaunch::new("tune-a", EngineKind::Llamacpp, &base)).is_err(), "adapter ids are taken too");

        sup.load("tune-a").await.unwrap();
        assert_eq!((sup.state("qwen"), sup.state("tune-a")), (Some(ModelState::Ready), Some(ModelState::Ready)), "loading an adapter loads its base");
        assert!(reply(&sup, "tune-a").await.unwrap().starts_with("Hello from qwen+ax0.5!"));
        assert!(reply(&sup, "qwen").await.unwrap().starts_with("Hello from qwen!"), "the base runs without the adapter");

        assert_eq!(sup.add_adapter("qwen", LoraAdapter::new("tune-b", &b)).unwrap(), AdapterChange::NextLoad);
        assert_eq!(sup.state("tune-b"), Some(ModelState::Unloaded));
        assert!(matches!(reply(&sup, "tune-b").await, Err(RemoteError::Unavailable { .. })));
        sup.load("tune-b").await.unwrap();
        assert!(reply(&sup, "tune-b").await.unwrap().starts_with("Hello from qwen+bx1.0!"), "the base restarted with it");
        assert!(reply(&sup, "tune-a").await.unwrap().starts_with("Hello from qwen+ax0.5!"));
        assert!(sup.add_adapter("qwen", LoraAdapter::new("tune-b", &b)).is_err(), "taken");
        assert!(sup.add_adapter("qwen", LoraAdapter::new("bad", &base)).is_err(), "not a LoRA");

        assert!(matches!(sup.unload("tune-a").await, Err(RemoteError::BadRequest { .. })));
        assert!(matches!(sup.remove_model("tune-a").await, Err(RemoteError::BadRequest { .. })));
        assert!(sup.embed("tune-a", &["x".into()]).await.is_err());
        sup.remove_adapter("qwen", "tune-b").unwrap();
        assert!(sup.state("tune-b").is_none());
        assert!(reply(&sup, "tune-b").await.is_err());

        let peft = tmp.path().join("vtune");
        std::fs::create_dir(&peft).unwrap();
        std::fs::write(peft.join("adapter_config.json"), br#"{"peft_type":"LORA"}"#).unwrap();
        sup.add_model(
            ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B").with_adapters(vec![LoraAdapter::new("vtune", peft.display().to_string())]),
        )
        .unwrap();
        sup.load("vtune").await.unwrap();
        assert!(reply(&sup, "vtune").await.unwrap().starts_with("Hello from vtune!"));
        assert!(reply(&sup, "big").await.unwrap().starts_with("Hello from big!"));

        sup.remove_model("qwen").await.unwrap();
        assert!(sup.state("tune-a").is_none(), "removing a base removes its adapters");
        // `big` started with an adapter, so LoRA is on and more go in at run time.
        assert_eq!(sup.add_adapter_live("big", LoraAdapter::new("vlive", "owner/live")).await.unwrap(), AdapterChange::Live);
        assert!(reply(&sup, "vlive").await.unwrap().starts_with("Hello from vlive!"));
        assert_eq!(sup.add_adapter("big", LoraAdapter::new("hub", "owner/lora")).unwrap(), AdapterChange::NextLoad);
        sup.unload("big").await.unwrap();
        assert_eq!(sup.add_adapter("big", LoraAdapter::new("hub2", "owner/lora2")).unwrap(), AdapterChange::Live, "the base isn't running");
        sup.shutdown().await;
    });
}

/// Adding, removing and rescaling adapters while the base keeps running: llama.cpp per-request
/// scales and re-adding a file it has loaded; vLLM run-time loading.
fn live_adapter_changes() {
    common::block_on(async {
        let tmp = tempfile::tempdir().unwrap();
        let w = |name: &str, b: Vec<u8>| {
            let p = tmp.path().join(name);
            std::fs::write(&p, b).unwrap();
            p.display().to_string()
        };
        let lora = || gguf("qwen3", &[("general.type", "adapter"), ("adapter.type", "lora")], &[]);
        let base = w("base.gguf", gguf("qwen3", &[], &[("qwen3.block_count", 1)]));
        let (a, b) = (w("a.gguf", lora()), w("b.gguf", lora()));
        let sup = supervisor(vec![]);
        sup.add_model(ModelLaunch::new("qwen", EngineKind::Llamacpp, &base).with_adapters(vec![LoraAdapter::new("tune-a", &a)])).unwrap();
        sup.load("qwen").await.unwrap();
        let pid = sup.status()[0].pid;
        let scaled = |s: Option<f32>| {
            let mut r = Request::messages(vec![Message::user("hi")]);
            if let Some(s) = s {
                r = r.with_adapter_scale(s);
            }
            r
        };
        assert!(text(&collect(&sup, "tune-a", &scaled(None)).await).starts_with("Hello from qwen+ax1.0!"));
        assert!(text(&collect(&sup, "tune-a", &scaled(Some(0.25))).await).starts_with("Hello from qwen+ax0.25!"), "per request");
        assert!(matches!(sup.chat("qwen", &scaled(Some(0.5))).await.err(), Some(RemoteError::BadRequest { .. })), "not an adapter");
        assert!(matches!(sup.chat("tune-a", &scaled(Some(f32::NAN))).await.err(), Some(RemoteError::BadRequest { .. })));
        let mut s = sup.complete_scaled("tune-a", "Tell me", &Sampling::default(), None, Some(0.5)).await.unwrap();
        while s.next().await.is_some() {}

        sup.set_adapter_scale("tune-a", Some(0.75)).unwrap();
        assert!(reply(&sup, "tune-a").await.unwrap().starts_with("Hello from qwen+ax0.75!"), "the new scale, same engine");
        assert!(sup.set_adapter_scale("qwen", Some(1.0)).is_err());
        assert!(sup.set_adapter_scale("tune-a", Some(100.0)).is_err());

        // Removed then added back (under another id too): the engine still has the file.
        sup.remove_adapter_live("qwen", "tune-a").await.unwrap();
        assert!(reply(&sup, "tune-a").await.is_err());
        assert_eq!(sup.add_adapter_live("qwen", LoraAdapter::new("tune-a2", &a).with_scale(0.5)).await.unwrap(), AdapterChange::Live);
        assert_eq!(sup.state("tune-a2"), Some(ModelState::Ready));
        assert!(reply(&sup, "tune-a2").await.unwrap().starts_with("Hello from qwen+ax0.5!"));
        // A new file can't go into a running llama-server.
        assert_eq!(sup.add_adapter_live("qwen", LoraAdapter::new("tune-b", &b)).await.unwrap(), AdapterChange::NextLoad);
        assert_eq!(sup.status()[0].pid, pid, "the base never restarted");

        // vLLM with run-time LoRA but no adapters at start.
        sup.add_model(ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B").with_lora_slots(2)).unwrap();
        sup.load("big").await.unwrap();
        let pid = sup.status().iter().find(|r| r.id == "big").unwrap().pid;
        assert_eq!(sup.add_adapter_live("big", LoraAdapter::new("v1", "owner/v1")).await.unwrap(), AdapterChange::Live);
        assert_eq!(sup.state("v1"), Some(ModelState::Ready));
        assert!(reply(&sup, "v1").await.unwrap().starts_with("Hello from v1!"));
        assert!(matches!(sup.chat("v1", &scaled(Some(0.5))).await.err(), Some(RemoteError::BadRequest { .. })), "vLLM takes no scale");
        assert!(sup.set_adapter_scale("v1", Some(0.5)).is_err());
        assert!(sup.add_adapter_live("big", LoraAdapter::new("v2", "owner/broken")).await.is_err(), "the engine refused it");
        assert!(sup.state("v2").is_none(), "and it was taken off again");
        sup.remove_adapter_live("big", "v1").await.unwrap();
        assert!(sup.state("v1").is_none());
        assert_eq!(
            sup.add_adapter_live("big", LoraAdapter::new("v1", "owner/v1")).await.unwrap(),
            AdapterChange::Live,
            "unloaded, so it loads again"
        );
        assert_eq!(sup.status().iter().find(|r| r.id == "big").unwrap().pid, pid, "the base never restarted");

        // Without run-time LoRA a vLLM base must restart.
        sup.add_model(ModelLaunch::new("plain", EngineKind::Vllm, "Qwen/Qwen3-4B")).unwrap();
        sup.load("plain").await.unwrap();
        assert_eq!(sup.add_adapter_live("plain", LoraAdapter::new("p1", "owner/p1")).await.unwrap(), AdapterChange::NextLoad);
        sup.shutdown().await;
    });
}

fn main() {
    common::main(&[
        ("lora_adapters_on_a_base", lora_adapters_on_a_base as fn()),
        ("live_adapter_changes", live_adapter_changes),
        ("mlx_behind_its_proxy", mlx_behind_its_proxy as fn()),
        ("chat_reasoning_tools_prefix_completion_and_embeddings", chat_reasoning_tools_prefix_completion_and_embeddings as fn()),
        ("swap_groups_and_idle_unload", swap_groups_and_idle_unload as fn()),
        ("models_added_and_removed_at_run_time", models_added_and_removed_at_run_time as fn()),
        ("wraps_a_loopback_server_and_refuses_others", wraps_a_loopback_server_and_refuses_others as fn()),
        #[cfg(unix)]
        ("crash_restarts_then_crash_loop", crash_restarts_then_crash_loop as fn()),
        ("raw_infer2_through_the_model_host", raw_infer2_through_the_model_host as fn()),
        ("images_and_audio_reach_llama_server_with_a_projector", images_and_audio_reach_llama_server_with_a_projector as fn()),
    ]);
}
