//! Engine request bodies (OpenAI `/v1/chat/completions` and `/v1/completions`) from infer/2
//! requests.

use serde_json::{Map, Value, json};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Input, Request, ResponseFormat, Sampling};
use weirdnet_agent::{Message, Role, ToolSpec};

use crate::engine::EngineKind;
use crate::error::bad_request;

/// An OpenAI chat message.
pub fn message(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        _ => "user",
    };
    let mut v = json!({"role": role, "content": m.content});
    if !m.tool_calls.is_empty() {
        v["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| {
                let args = match &c.arguments {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": args}})
            })
            .collect();
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = id.clone().into();
    }
    v
}

/// An OpenAI function tool.
pub fn tool(t: &ToolSpec) -> Value {
    json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}})
}

fn sampling(obj: &mut Map<String, Value>, s: &Sampling, engine: EngineKind) {
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            obj.insert(k.into(), v);
        }
    };
    put("temperature", s.temperature.map(Value::from));
    put("top_p", s.top_p.map(Value::from));
    put("top_k", s.top_k.map(Value::from));
    put("min_p", s.min_p.map(Value::from));
    put("max_tokens", s.max_tokens.map(Value::from));
    put("presence_penalty", s.presence_penalty.map(Value::from));
    put("frequency_penalty", s.frequency_penalty.map(Value::from));
    let repeat = if engine == EngineKind::Vllm { "repetition_penalty" } else { "repeat_penalty" };
    put(repeat, s.repeat_penalty.map(Value::from));
    if !s.stop.is_empty() {
        obj.insert("stop".into(), s.stop.clone().into());
    }
}

fn streaming(obj: &mut Map<String, Value>, alias: &str) {
    obj.insert("model".into(), alias.into());
    obj.insert("stream".into(), true.into());
    obj.insert("stream_options".into(), json!({"include_usage": true}));
}

/// The streaming chat body for `request` on the engine's `alias`. An assistant prefix becomes a
/// final assistant message the engine continues (llama.cpp prefills a trailing assistant
/// message; vLLM gets `continue_final_message`). Reasoning in the prefix is not continued.
pub fn chat_body(request: &Request, alias: &str, engine: EngineKind) -> Result<Value, RemoteError> {
    let Input::Messages(messages) = &request.input else {
        return Err(bad_request("chat takes assembled messages, not a context_ref or a raw prompt"));
    };
    let mut msgs: Vec<Value> = messages.iter().map(message).collect();
    let prefix = request.assistant_prefix.as_ref().filter(|p| !p.text.is_empty());
    if let Some(p) = prefix {
        msgs.push(json!({"role": "assistant", "content": p.text}));
    }
    if msgs.is_empty() {
        return Err(bad_request("no messages"));
    }
    let mut obj = Map::new();
    obj.insert("messages".into(), Value::Array(msgs));
    if prefix.is_some() && engine == EngineKind::Vllm {
        obj.insert("continue_final_message".into(), true.into());
        obj.insert("add_generation_prompt".into(), false.into());
    }
    sampling(&mut obj, &request.sampling, engine);
    if let Some(seed) = request.seed {
        obj.insert("seed".into(), seed.into());
    }
    match &request.response_format {
        Some(ResponseFormat::Json) => {
            obj.insert("response_format".into(), json!({"type": "json_object"}));
        }
        Some(ResponseFormat::JsonSchema { name, schema, strict }) => {
            obj.insert("response_format".into(), json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}}));
        }
        _ => {}
    }
    if !request.tools.is_empty() {
        obj.insert("tools".into(), request.tools.iter().map(tool).collect());
    }
    streaming(&mut obj, alias);
    Ok(Value::Object(obj))
}

/// The streaming `/v1/completions` body for a raw `prompt`.
pub fn completion_body(prompt: &str, s: &Sampling, seed: Option<u64>, alias: &str, engine: EngineKind) -> Value {
    let mut obj = Map::new();
    obj.insert("prompt".into(), prompt.into());
    sampling(&mut obj, s, engine);
    if let Some(seed) = seed {
        obj.insert("seed".into(), seed.into());
    }
    streaming(&mut obj, alias);
    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_agent::ToolCall;
    use weirdnet_agent::remote::infer::v2::{ContextRef, Prefix};

    fn req() -> Request {
        let call = Message::assistant("", vec![ToolCall { id: "c1".into(), name: "f".into(), arguments: json!({"x": 1}) }]);
        let result = Message::tool("c1", "42");
        let mut s = Sampling::default();
        s.temperature = Some(0.0);
        s.max_tokens = Some(5);
        s.repeat_penalty = Some(1.1);
        s.stop = vec!["\n".into()];
        Request::messages(vec![Message::system("be brief"), Message::user("hi"), call, result])
            .with_sampling(s)
            .with_seed(7)
            .with_tools(vec![ToolSpec::new("f", "a tool")])
            .with_response_format(ResponseFormat::Json)
    }

    #[test]
    fn chat_bodies() {
        let b = chat_body(&req(), "alias", EngineKind::Llamacpp).unwrap();
        assert_eq!(b["model"], "alias");
        assert_eq!(b["stream"], true);
        assert_eq!(b["stream_options"]["include_usage"], true);
        assert_eq!(b["messages"][0], json!({"role": "system", "content": "be brief"}));
        assert_eq!(b["messages"][2]["tool_calls"][0]["function"]["arguments"], "{\"x\":1}");
        assert_eq!(b["messages"][3], json!({"role": "tool", "content": "42", "tool_call_id": "c1"}));
        assert_eq!((b["temperature"].clone(), b["max_tokens"].clone(), b["seed"].clone()), (json!(0.0), json!(5), json!(7)));
        assert!(b["repeat_penalty"].is_number() && b.get("repetition_penalty").is_none());
        assert_eq!(b["stop"], json!(["\n"]));
        assert_eq!(b["tools"][0]["function"]["name"], "f");
        assert_eq!(b["response_format"]["type"], "json_object");
        let v = chat_body(&req(), "alias", EngineKind::Vllm).unwrap();
        assert!(v["repetition_penalty"].is_number());
        assert!(v.get("continue_final_message").is_none());
    }

    #[test]
    fn a_prefix_is_continued() {
        let r = Request::messages(vec![Message::user("hi")]).with_prefix(Prefix::new("Hello th", ""));
        let b = chat_body(&r, "m", EngineKind::Llamacpp).unwrap();
        assert_eq!(b["messages"][1], json!({"role": "assistant", "content": "Hello th"}));
        let v = chat_body(&r, "m", EngineKind::Vllm).unwrap();
        assert_eq!((v["continue_final_message"].clone(), v["add_generation_prompt"].clone()), (json!(true), json!(false)));
    }

    #[test]
    fn context_refs_and_empty_requests_are_refused() {
        assert!(matches!(chat_body(&Request::context(ContextRef::new("c", None)), "m", EngineKind::Llamacpp), Err(RemoteError::BadRequest { .. })));
        assert!(chat_body(&Request::messages(vec![]), "m", EngineKind::Llamacpp).is_err());
    }

    #[test]
    fn completion_bodies() {
        let b = completion_body("Once", &Sampling::default(), None, "m", EngineKind::Llamacpp);
        assert_eq!((b["prompt"].clone(), b["stream"].clone()), (json!("Once"), json!(true)));
    }
}
