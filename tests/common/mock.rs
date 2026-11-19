//! A stand-in for `llama-server` and `vllm serve`: the same flags and endpoints, canned answers.
//! Test binaries run themselves as the fake engine when started with `--port`.
//!
//! Chat answers "Hello from <alias>! You said: <last user message>" word by word. A last
//! message from the assistant is a prefix: the answer continues it with "world". A user
//! message starting with "think" first streams reasoning. With tools, the answer is a call to
//! the first tool with `{"q": "<message>"}`, split over two chunks. `/health` says 503 for
//! `WEIRDNET_MOCK_LOAD_MS` (default 200) after start. `/v1/completions` continues the prompt
//! with " once upon a time"; `/v1/embeddings` returns `[chars, words, 1, 0]` per input.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};

struct Mock {
    alias: String,
    key: Option<String>,
    ready_at: Instant,
}

/// Runs the mock engine (the test binary re-executes itself with engine flags).
pub fn main() {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(serve());
}

async fn serve() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let host = flag("--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let alias = flag("--alias").or_else(|| flag("--served-model-name")).or_else(|| flag("--model")).unwrap_or_else(|| "mock".into());
    let load_ms = std::env::var("WEIRDNET_MOCK_LOAD_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    assert_eq!(host, "127.0.0.1", "engines must bind loopback only");
    let key = std::env::var("LLAMA_API_KEY").or_else(|_| std::env::var("VLLM_API_KEY")).ok();
    let st = Arc::new(Mock { alias, key, ready_at: Instant::now() + Duration::from_millis(load_ms) });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(st);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.expect("bind");
    axum::serve(listener, app).await.expect("serve");
}

async fn health(State(st): State<Arc<Mock>>) -> Response {
    if Instant::now() < st.ready_at {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"error": {"message": "Loading model", "code": 503}}))).into_response();
    }
    axum::Json(json!({"status": "ok"})).into_response()
}

fn check(st: &Mock, headers: &HeaderMap, body: &Value) -> Option<Response> {
    let authorized = match &st.key {
        None => true,
        Some(k) => headers.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {k}")),
    };
    if !authorized {
        return Some((StatusCode::UNAUTHORIZED, axum::Json(json!({"error": {"message": "Invalid API Key"}}))).into_response());
    }
    if body["model"].as_str() != Some(st.alias.as_str()) {
        let msg = format!("unknown model {}", body["model"]);
        return Some((StatusCode::BAD_REQUEST, axum::Json(json!({"error": {"message": msg}}))).into_response());
    }
    None
}

fn sse(events: Vec<Value>) -> Response {
    let stream = futures::stream::unfold((events.into_iter(), false), |(mut it, done)| async move {
        if done {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
        match it.next() {
            Some(v) => Some((Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string())), (it, false))),
            None => Some((Ok(Event::default().data("[DONE]")), (it, true))),
        }
    });
    Sse::new(stream).into_response()
}

async fn completions(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if let Some(r) = check(&st, &headers, &body) {
        return r;
    }
    let mut events: Vec<Value> = [" once", " upon", " a", " time"]
        .iter()
        .map(|w| json!({"object": "text_completion", "choices": [{"index": 0, "text": w, "finish_reason": null}]}))
        .collect();
    events.push(json!({"object": "text_completion", "choices": [{"index": 0, "text": "", "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}));
    sse(events)
}

async fn embeddings(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if let Some(r) = check(&st, &headers, &body) {
        return r;
    }
    let inputs: Vec<String> = match &body["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"object": "embedding", "index": i, "embedding": [s.chars().count() as f32, s.split_whitespace().count() as f32, 1.0, 0.0]}))
        .collect();
    axum::Json(json!({"object": "list", "data": data, "usage": {"prompt_tokens": inputs.len(), "total_tokens": inputs.len()}})).into_response()
}

async fn chat(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if let Some(r) = check(&st, &headers, &body) {
        return r;
    }
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    let said = messages.iter().rev().find(|m| m["role"] == "user").and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
    let chunk = |delta: Value, finish: Option<&str>| json!({"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    let mut events = vec![chunk(json!({"role": "assistant", "content": null}), None)];
    let mut n = 0u64;
    let finish;
    if let Some(tool) = body["tools"].as_array().and_then(|t| t.first()) {
        let name = tool["function"]["name"].as_str().unwrap_or("tool");
        let args = json!({"q": said}).to_string();
        let (a, b) = args.split_at(args.len() / 2);
        events
            .push(chunk(json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": name, "arguments": a}}]}), None));
        events.push(chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": b}}]}), None));
        n = 2;
        finish = "tool_calls";
    } else {
        if said.starts_with("think") {
            events.push(chunk(json!({"reasoning_content": "hmm"}), None));
            n += 1;
        }
        let continuing = messages.last().is_some_and(|m| m["role"] == "assistant");
        let answer = if continuing { "world".to_string() } else { format!("Hello from {}! You said: {said}", st.alias) };
        for w in answer.split_inclusive(' ') {
            events.push(chunk(json!({"content": w}), None));
            n += 1;
        }
        finish = "stop";
    }
    events.push(chunk(json!({}), Some(finish)));
    events.push(
        json!({"object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n}}),
    );
    sse(events)
}
