//! MLX-LM (`mlx_lm.server`) on Apple silicon: installing a pinned `mlx-lm`, and the proxy the
//! supervisor puts in front of it.
//!
//! MLX-LM's server differs from llama.cpp and vLLM in three ways the supervisor hides:
//!
//! - It takes no API key. The supervisor starts it on a private loopback port and serves the
//!   model's advertised port from a small proxy that checks the per-process key as a bearer token.
//! - Its `/health` answers 200 before the weights are loaded. The proxy's `/health` turns 200 only
//!   after a one-token completion has succeeded.
//! - It serves the model it was started with as `default_model`. The proxy rewrites each request's
//!   `model` to that, lists the pool id in `/v1/models`, refuses image and audio content parts and
//!   `/v1/embeddings` with 400 (it is text only here and has no embeddings), and streams the rest
//!   through.
//!
//! Install: [`install`] makes a venv with `uv` (which fetches its own Python) when it is on
//! `PATH`, else `python3 -m venv`, and installs `mlx-lm==`[`MLX_LM_VERSION`] into it; point
//! [`EngineConfig::mlx`](crate::EngineConfig::mlx) at [`server_in`] of that venv.
//!
//! MLX runs on macOS on Apple silicon only ([`supported`]).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::{Value, json};

use crate::engine::find_in_path;

/// The `mlx-lm` release [`install`] pins.
pub const MLX_LM_VERSION: &str = "0.32.2";
/// The Python `uv` asks for when it makes the venv.
pub const PYTHON: &str = "3.12";
/// `--max-tokens` when the model has no context set (MLX-LM's own default is 512).
pub const DEFAULT_MAX_TOKENS: u32 = 32768;
/// The model name MLX-LM serves the model it was started with under.
pub const DEFAULT_MODEL: &str = "default_model";

/// MLX runs on macOS on Apple silicon only (`std::env::consts` names).
pub fn supported(os: &str, arch: &str) -> bool {
    os == "macos" && arch == "aarch64"
}

/// Whether this build runs where MLX does.
pub fn supported_here() -> bool {
    supported(std::env::consts::OS, std::env::consts::ARCH)
}

/// `mlx_lm.server` in a venv.
pub fn server_in(venv: &Path) -> PathBuf {
    venv.join("bin").join("mlx_lm.server")
}

/// The commands that install the pinned `mlx-lm` into `venv`: with `uv` if given, else with
/// `python3 -m venv` and pip.
pub fn install_commands(venv: &Path, uv: Option<&Path>) -> Vec<Command> {
    let spec = format!("mlx-lm=={MLX_LM_VERSION}");
    let py = venv.join("bin").join("python");
    match uv {
        Some(uv) => {
            let mut a = Command::new(uv);
            a.args(["venv", "--python", PYTHON]).arg(venv);
            let mut b = Command::new(uv);
            b.args(["pip", "install", "--python"]).arg(&py).arg(&spec);
            vec![a, b]
        }
        None => {
            let mut a = Command::new("python3");
            a.args(["-m", "venv"]).arg(venv);
            let mut b = Command::new(&py);
            b.args(["-m", "pip", "install", "--upgrade", "pip"]);
            let mut c = Command::new(&py);
            c.args(["-m", "pip", "install"]).arg(&spec);
            vec![a, b, c]
        }
    }
}

/// Installs the pinned `mlx-lm` into `venv` (blocking; it downloads) and returns its
/// `mlx_lm.server`. Needs `uv` or `python3` on `PATH`, and a Mac with Apple silicon.
pub fn install(venv: &Path) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    if !supported_here() {
        return Err(Error::new(ErrorKind::Unsupported, "MLX-LM runs on macOS on Apple silicon only"));
    }
    let uv = find_in_path("uv");
    if uv.is_none() && find_in_path("python3").is_none() {
        return Err(Error::new(ErrorKind::NotFound, "installing MLX-LM needs `uv` (https://docs.astral.sh/uv/) or `python3` on PATH"));
    }
    if let Some(parent) = venv.parent() {
        std::fs::create_dir_all(parent)?;
    }
    for mut cmd in install_commands(venv, uv.as_deref()) {
        let shown = format!("{cmd:?}");
        let status = cmd.status().map_err(|e| Error::new(e.kind(), format!("running {shown}: {e}")))?;
        if !status.success() {
            return Err(Error::other(format!("{shown} failed ({status})")));
        }
    }
    let server = server_in(venv);
    if !server.is_file() {
        return Err(Error::new(ErrorKind::NotFound, format!("mlx-lm installed but {} is missing", server.display())));
    }
    Ok(server)
}

/// Image or audio content parts in a chat request.
pub fn has_media(body: &Value) -> bool {
    body["messages"].as_array().is_some_and(|ms| {
        ms.iter().any(|m| {
            m["content"].as_array().is_some_and(|parts| {
                parts.iter().any(|p| matches!(p["type"].as_str(), Some("image_url" | "input_image" | "image" | "input_audio" | "audio")))
            })
        })
    })
}

/// Request bodies can be large (long prompts).
const MAX_BODY: usize = 256 << 20;

struct Proxy {
    upstream: String,
    alias: String,
    key: String,
    http: reqwest::Client,
    ready: AtomicBool,
}

/// The proxy in front of one `mlx_lm.server`; aborts its tasks when dropped.
pub(crate) struct ProxyHandle(tokio::task::JoinHandle<()>);

impl Drop for ProxyHandle {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Serves `listener` (the model's advertised port) in front of the MLX-LM server at `upstream`,
/// accepting only `key` as the bearer token, and warms the model.
pub(crate) fn spawn_proxy(listener: tokio::net::TcpListener, upstream: String, alias: String, key: String) -> ProxyHandle {
    let http =
        reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).no_proxy().build().expect("an HTTP client without TLS always builds");
    let st = Arc::new(Proxy { upstream, alias, key, http, ready: AtomicBool::new(false) });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", axum::routing::any(no_embeddings))
        .fallback(forward)
        .with_state(st.clone());
    ProxyHandle(tokio::spawn(async move {
        tokio::join!(warm(st), async {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::warn!("mlx proxy stopped: {e}");
            }
        });
    }))
}

fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, axum::Json(json!({"error": {"code": status.as_u16(), "message": message, "type": kind}}))).into_response()
}

/// Ready means the model is loaded and has produced a token, not just that the port is open.
async fn warm(st: Arc<Proxy>) {
    let chat = json!({"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": false});
    let raw = json!({"model": DEFAULT_MODEL, "prompt": "hi", "max_tokens": 1, "stream": false});
    loop {
        match status(&st, "/v1/chat/completions", &chat).await {
            Some(s) if s.is_success() => break,
            // No chat template: a raw completion proves the model is loaded just as well.
            Some(s) if s.is_client_error() && status(&st, "/v1/completions", &raw).await.is_some_and(|s| s.is_success()) => break,
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    st.ready.store(true, Ordering::SeqCst);
    tracing::info!(model = %st.alias, "mlx model warm");
}

async fn status(st: &Proxy, path: &str, body: &Value) -> Option<reqwest::StatusCode> {
    let r = st.http.post(format!("{}{path}", st.upstream)).json(body).timeout(Duration::from_secs(600)).send().await.ok()?;
    let s = r.status();
    let _ = r.bytes().await;
    Some(s)
}

async fn health(State(st): State<Arc<Proxy>>) -> Response {
    if st.ready.load(Ordering::SeqCst) {
        axum::Json(json!({"status": "ok"})).into_response()
    } else {
        error(StatusCode::SERVICE_UNAVAILABLE, "unavailable_error", "Loading model")
    }
}

fn deny(st: &Proxy, headers: &HeaderMap) -> Option<Response> {
    let given = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    (!same(given.as_bytes(), st.key.as_bytes())).then(|| error(StatusCode::UNAUTHORIZED, "authentication_error", "Invalid API Key"))
}

/// Constant-time comparison (for equal lengths).
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn models(State(st): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
    if let Some(r) = deny(&st, &headers) {
        return r;
    }
    axum::Json(json!({"object": "list", "data": [{"id": st.alias, "object": "model", "owned_by": "weirdnet"}]})).into_response()
}

async fn no_embeddings(State(st): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
    if let Some(r) = deny(&st, &headers) {
        return r;
    }
    error(StatusCode::BAD_REQUEST, "invalid_request_error", "this model runs on MLX-LM, which serves no embeddings")
}

async fn forward(State(st): State<Arc<Proxy>>, req: Request) -> Response {
    if let Some(r) = deny(&st, req.headers()) {
        return r;
    }
    let method = req.method().clone();
    let path = req.uri().path_and_query().map_or("/", |p| p.as_str()).to_string();
    let ctype = req.headers().get(header::CONTENT_TYPE).cloned();
    let bytes = match axum::body::to_bytes(req.into_body(), MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", &e.to_string()),
    };
    let body = match serde_json::from_slice::<Value>(&bytes) {
        Ok(mut v) if v.is_object() => {
            if has_media(&v) {
                return error(StatusCode::BAD_REQUEST, "invalid_request_error", "this model runs on MLX-LM, which takes text only");
            }
            if v.get("model").is_some() {
                v["model"] = Value::String(DEFAULT_MODEL.into());
            }
            serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec())
        }
        _ => bytes.to_vec(),
    };
    let method = reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::POST);
    let mut up = st.http.request(method, format!("{}{path}", st.upstream)).body(body);
    if let Some(c) = ctype.as_ref().and_then(|c| c.to_str().ok()) {
        up = up.header(reqwest::header::CONTENT_TYPE, c);
    }
    match up.send().await {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ctype = r.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok());
            let mut resp = Response::new(Body::from_stream(r.bytes_stream()));
            *resp.status_mut() = status;
            if let Some(c) = ctype {
                resp.headers_mut().insert(header::CONTENT_TYPE, c);
            }
            resp
        }
        Err(e) => error(StatusCode::BAD_GATEWAY, "engine_error", &format!("engine: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use axum::routing::post;

    use super::*;

    #[test]
    fn install_uses_uv_when_present() {
        let c = install_commands(Path::new("/h/engines/mlx"), Some(Path::new("/usr/bin/uv")));
        let args: Vec<Vec<String>> = c.iter().map(|c| c.get_args().map(|a| a.to_string_lossy().into_owned()).collect()).collect();
        assert_eq!(args[0], ["venv", "--python", PYTHON, "/h/engines/mlx"]);
        assert_eq!(args[1][..3], ["pip", "install", "--python"]);
        assert_eq!(args[1].last().unwrap(), &format!("mlx-lm=={MLX_LM_VERSION}"));
        let p = install_commands(Path::new("/v"), None);
        assert_eq!(p[0].get_program(), "python3");
        assert_eq!(p.len(), 3);
        assert_eq!(server_in(Path::new("/v")), Path::new("/v/bin/mlx_lm.server"));
    }

    #[test]
    fn platforms() {
        assert!(supported("macos", "aarch64"));
        assert!(!supported("macos", "x86_64"));
        assert!(!supported("linux", "aarch64"));
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert_eq!(install(Path::new("/nonexistent/mlx")).unwrap_err().kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn media_parts() {
        let img = json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "?"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]});
        assert!(has_media(&img));
        assert!(has_media(
            &json!({"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "", "format": "wav"}}]}]})
        ));
        assert!(!has_media(&json!({"messages": [{"role": "user", "content": "hi"}]})));
        assert!(!has_media(&json!({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})));
        assert!(same(b"abc", b"abc") && !same(b"abc", b"abd") && !same(b"abc", b"ab"));
    }

    /// A fake `mlx_lm.server`: `/health` is 200 at once, completions fail until `load` requests
    /// have been made, and every request must name `default_model`.
    #[derive(Default)]
    struct Fake {
        attempts: AtomicUsize,
        load: usize,
        seen: Mutex<Vec<Value>>,
    }

    async fn fake_chat(State(f): State<Arc<Fake>>, axum::Json(body): axum::Json<Value>) -> Response {
        f.seen.lock().unwrap().push(body.clone());
        if f.attempts.fetch_add(1, Ordering::SeqCst) < f.load {
            return (StatusCode::INTERNAL_SERVER_ERROR, "loading").into_response();
        }
        if body["model"] != DEFAULT_MODEL {
            return (StatusCode::NOT_FOUND, "unknown model").into_response();
        }
        axum::Json(json!({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})).into_response()
    }

    async fn fake(load: usize) -> (String, Arc<Fake>) {
        let f = Arc::new(Fake { load, ..Default::default() });
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/chat/completions", post(fake_chat))
            .route("/v1/completions", post(fake_chat))
            .with_state(f.clone());
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        (url, f)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn proxy_warms_checks_the_key_and_rewrites() {
        let (upstream, f) = fake(3).await;
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let _h = spawn_proxy(l, upstream, "qwen3-8b".into(), "secret".into());
        let http = reqwest::Client::new();

        // Not ready until a completion has gone through, though the engine's /health says 200.
        let health = http.get(format!("{base}/health")).send().await.unwrap();
        assert_eq!(health.status(), 503);
        let mut ready = false;
        for _ in 0..100 {
            if http.get(format!("{base}/health")).send().await.unwrap().status() == 200 {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(ready, "warm never finished");
        assert!(f.attempts.load(Ordering::SeqCst) > 3);

        let chat = json!({"model": "qwen3-8b", "messages": [{"role": "user", "content": "hi"}]});
        let url = format!("{base}/v1/chat/completions");
        assert_eq!(http.post(&url).json(&chat).send().await.unwrap().status(), 401);
        assert_eq!(http.post(&url).bearer_auth("wrong!").json(&chat).send().await.unwrap().status(), 401);
        assert_eq!(http.get(format!("{base}/v1/models")).send().await.unwrap().status(), 401);

        let r = http.post(&url).bearer_auth("secret").json(&chat).send().await.unwrap();
        assert_eq!(r.status(), 200);
        assert_eq!(r.json::<Value>().await.unwrap()["choices"][0]["message"]["content"], "ok");
        assert_eq!(f.seen.lock().unwrap().last().unwrap()["model"], DEFAULT_MODEL);

        let m: Value = http.get(format!("{base}/v1/models")).bearer_auth("secret").send().await.unwrap().json().await.unwrap();
        assert_eq!(m["data"][0]["id"], "qwen3-8b");

        let n = f.seen.lock().unwrap().len();
        let img = json!({"model": "qwen3-8b", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]});
        assert_eq!(http.post(&url).bearer_auth("secret").json(&img).send().await.unwrap().status(), 400);
        let emb =
            http.post(format!("{base}/v1/embeddings")).bearer_auth("secret").json(&json!({"model": "qwen3-8b", "input": "x"})).send().await.unwrap();
        assert_eq!(emb.status(), 400);
        assert_eq!(f.seen.lock().unwrap().len(), n, "refused requests never reach the engine");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dropping_the_handle_stops_the_proxy() {
        let (upstream, _f) = fake(0).await;
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        let h = spawn_proxy(l, upstream, "m".into(), "k".into());
        let http = reqwest::Client::builder().no_proxy().pool_max_idle_per_host(0).build().unwrap();
        assert!(http.get(format!("{base}/health")).send().await.is_ok());
        drop(h);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(http.get(format!("{base}/health")).send().await.is_err());
    }
}
