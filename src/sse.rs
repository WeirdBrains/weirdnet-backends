//! OpenAI-style server-sent events from an engine, decoded into [`Chunk`]s.

use serde_json::Value;
use weirdnet_agent::ToolCall;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::Usage;

use crate::error::engine_error;

/// Splits a byte stream into SSE `data:` payloads.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
    data: String,
}

impl SseParser {
    /// Feeds bytes; returns the payloads of every event completed by them.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(std::mem::take(&mut self.data));
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            }
        }
        out
    }

    /// Whatever is left when the stream ends without a blank line.
    pub fn finish(&mut self) -> Option<String> {
        let rest = self.feed(b"\n\n");
        rest.into_iter().next()
    }
}

/// One decoded piece of an engine's stream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Chunk {
    Text(String),
    Reasoning(String),
    /// A fragment of a streamed tool call: the first fragment for an `index` has the id and name,
    /// later ones add to the arguments. [`ToolCalls`] puts them together.
    ToolCall {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: String,
    },
    Done {
        finish_reason: Option<String>,
        usage: Option<Usage>,
    },
}

/// Turns OpenAI `chat.completion.chunk` (and `text_completion`) JSON into [`Chunk`]s.
#[derive(Debug, Default)]
pub struct ChunkDecoder {
    finish_reason: Option<String>,
    usage: Option<Usage>,
    done: bool,
}

impl ChunkDecoder {
    /// Decodes one payload. `[DONE]` yields the final `Done`.
    pub fn decode(&mut self, payload: &str) -> Result<Vec<Chunk>, RemoteError> {
        if payload.trim() == "[DONE]" {
            return Ok(self.finish().into_iter().collect());
        }
        let v: Value = serde_json::from_str(payload).map_err(|e| engine_error(format!("bad chunk: {e}")))?;
        if let Some(err) = v.get("error") {
            let msg = err.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| err.to_string());
            return Err(engine_error(msg));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            let mut usage = Usage::new(
                u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
            );
            usage.cached_tokens = u.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0);
            usage.reasoning_tokens = u.pointer("/completion_tokens_details/reasoning_tokens").and_then(Value::as_u64).unwrap_or(0);
            self.usage = Some(usage);
        }
        let mut out = Vec::new();
        let Some(choice) = v.get("choices").and_then(Value::as_array).and_then(|c| c.first()) else { return Ok(out) };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        // `/v1/completions` chunks carry `text` instead of a delta.
        if let Some(text) = choice.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            out.push(Chunk::Text(text.to_string()));
        }
        let Some(delta) = choice.get("delta") else { return Ok(out) };
        for key in ["reasoning_content", "reasoning"] {
            if let Some(text) = delta.get(key).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                out.push(Chunk::Reasoning(text.to_string()));
                break;
            }
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            out.push(Chunk::Text(text.to_string()));
        }
        for (i, call) in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let f = call.get("function");
            out.push(Chunk::ToolCall {
                index: call.get("index").and_then(Value::as_u64).unwrap_or(i as u64) as u32,
                id: call.get("id").and_then(Value::as_str).map(str::to_string),
                name: f.and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string),
                arguments: f.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("").to_string(),
            });
        }
        Ok(out)
    }

    /// The `Done`, once.
    pub fn finish(&mut self) -> Option<Chunk> {
        if self.done {
            return None;
        }
        self.done = true;
        Some(Chunk::Done { finish_reason: self.finish_reason.take(), usage: self.usage })
    }

    /// The engine sent a finish reason or `[DONE]`.
    pub fn complete(&self) -> bool {
        self.done || self.finish_reason.is_some()
    }
}

/// Puts streamed tool-call fragments together into whole calls, in index order.
#[derive(Debug, Default)]
pub struct ToolCalls {
    calls: Vec<(u32, Option<String>, Option<String>, String)>,
}

impl ToolCalls {
    pub fn push(&mut self, index: u32, id: Option<String>, name: Option<String>, arguments: &str) {
        match self.calls.iter_mut().find(|c| c.0 == index) {
            Some(c) => {
                if c.1.is_none() {
                    c.1 = id;
                }
                if c.2.is_none() {
                    c.2 = name;
                }
                c.3.push_str(arguments);
            }
            None => self.calls.push((index, id, name, arguments.to_string())),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// The whole calls. Arguments that are not JSON are kept as a string; a call without an id
    /// gets `call_<index>`.
    pub fn finish(mut self) -> Vec<ToolCall> {
        self.calls.sort_by_key(|c| c.0);
        self.calls
            .into_iter()
            .map(|(index, id, name, args)| ToolCall {
                id: id.unwrap_or_else(|| format!("call_{index}")),
                name: name.unwrap_or_default(),
                arguments: if args.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&args).unwrap_or(Value::String(args))
                },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_events() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: {\"a\":").is_empty());
        assert_eq!(p.feed(b"1}\r\n\r\ndata: [DONE]\n\n"), vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn decodes_llama_chunks() {
        let mut d = ChunkDecoder::default();
        assert!(d.decode(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}"#).unwrap().is_empty());
        assert_eq!(d.decode(r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#).unwrap(), vec![Chunk::Text("Hello".into())]);
        assert_eq!(d.decode(r#"{"choices":[{"index":0,"delta":{"reasoning_content":"hm"}}]}"#).unwrap(), vec![Chunk::Reasoning("hm".into())]);
        d.decode(r#"{"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}"#).unwrap();
        d.decode(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":1}}}"#).unwrap();
        let mut usage = Usage::new(3, 2);
        usage.cached_tokens = 1;
        assert_eq!(d.decode("[DONE]").unwrap(), vec![Chunk::Done { finish_reason: Some("stop".into()), usage: Some(usage) }]);
        assert!(d.finish().is_none());
    }

    #[test]
    fn completion_chunks() {
        let mut d = ChunkDecoder::default();
        assert_eq!(d.decode(r#"{"choices":[{"index":0,"text":"Once"}]}"#).unwrap(), vec![Chunk::Text("Once".into())]);
    }

    #[test]
    fn tool_calls_and_errors() {
        let mut d = ChunkDecoder::default();
        let mut calls = ToolCalls::default();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"x\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"name":"g","arguments":"not json"}}]}}]}"#,
        ] {
            for c in d.decode(payload).unwrap() {
                if let Chunk::ToolCall { index, id, name, arguments } = c {
                    calls.push(index, id, name, &arguments);
                }
            }
        }
        let calls = calls.finish();
        assert_eq!((calls[0].id.as_str(), calls[0].name.as_str()), ("c1", "f"));
        assert_eq!(calls[0].arguments, serde_json::json!({"x": 1}));
        assert_eq!((calls[1].id.as_str(), calls[1].arguments.clone()), ("call_1", Value::String("not json".into())));
        assert!(d.decode(r#"{"error":{"message":"boom"}}"#).is_err());
    }
}
