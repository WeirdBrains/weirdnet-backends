//! Speech out on the engines: llama-server's and vLLM-Omni's `/v1/audio/speech` request bodies
//! from infer/2 `Speak` requests, the speech profile each engine reports at load, and what an
//! answer's headers say about its audio. Each adapter maps only what the model does (its
//! [`SpeechProfile`]); nothing is faked, and a reference clip only ever travels inline.

use base64::Engine as _;
use serde_json::{Map, Value, json};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Meta, SpeakInput, SpeakRequest, Voice};
use weirdnet_agent::speech::{AudioFormat, SpeechProfile};

use crate::engine::{EngineKind, ModelLaunch};
use crate::error::bad_request;

/// One engine request: the JSON body, and what the answer will be.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeechBody {
    pub body: Value,
    pub format: AudioFormat,
    /// The engine streams the audio as it makes it.
    pub stream: bool,
    /// Request fields the model received but doesn't use (`Meta::ignored`).
    pub meta: Meta,
    /// Channels of headerless PCM when the engine documents them (llama-server: mono).
    pub pcm_channels: Option<u16>,
}

/// The engine's name for `format` in `response_format`.
fn format_name(engine: EngineKind, model: &str, format: AudioFormat) -> Result<&'static str, RemoteError> {
    match (engine, format) {
        (_, AudioFormat::Wav) => Ok("wav"),
        (_, AudioFormat::Pcm16) => Ok("pcm"),
        (EngineKind::VllmOmni, AudioFormat::Mp3) => Ok("mp3"),
        (EngineKind::VllmOmni, AudioFormat::Flac) => Ok("flac"),
        (EngineKind::VllmOmni, AudioFormat::Opus) => Ok("opus"),
        (EngineKind::Llamacpp, other) => Err(bad_request(format!("`{model}` runs on llama-server, which makes wav and pcm16 only, not `{other}`"))),
        (other, _) => Err(bad_request(format!("`{model}` runs on {}, which serves no speech", other.name()))),
    }
}

/// A multi-speaker script as speaker-tagged text (`<|speaker:N|>` before each turn, one turn a
/// line), as Fish Audio Drama 3 takes it on both engines.
pub(crate) fn script_text(input: &SpeakInput) -> Option<String> {
    match input {
        SpeakInput::Script(turns) => Some(turns.iter().map(|t| format!("<|speaker:{}|>{}", t.speaker, t.text.trim())).collect::<Vec<_>>().join("\n")),
        _ => None,
    }
}

/// The MIME type of a reference clip in `format` (`wav`, `mp3`, ...). A format must be a plain
/// name: nothing that could be read as a path or URL is ever put in front of an engine.
pub(crate) fn clip_mime(format: &str) -> Result<String, RemoteError> {
    let f = format.trim().to_ascii_lowercase();
    if f.is_empty() || f.len() > 16 || !f.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(bad_request("a reference clip's format must be a plain name such as wav or mp3"));
    }
    Ok(match f.as_str() {
        "wav" | "wave" => "audio/wav".into(),
        "mp3" | "mpeg" => "audio/mpeg".into(),
        "ogg" | "opus" => "audio/ogg".into(),
        "m4a" | "mp4" | "aac" => "audio/mp4".into(),
        other => format!("audio/{other}"),
    })
}

/// vLLM-Omni's `cache_salt` for a clone request: a hash of the request id, so no conditioning
/// prefix is shared across callers.
pub(crate) fn cache_salt(id: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    "weirdnet-backends/speak".hash(&mut h);
    id.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// vLLM-Omni's `task_type` for a Qwen3-TTS checkpoint (`CustomVoice`, `VoiceDesign`, `Base`),
/// read from the launch's id or source; `None` for other models.
pub(crate) fn task_type(launch: &ModelLaunch) -> Option<&'static str> {
    if launch.engine != EngineKind::VllmOmni {
        return None;
    }
    let names = [launch.id.to_ascii_lowercase(), launch.source.to_ascii_lowercase()];
    let qwen = names.iter().any(|n| n.contains("qwen3-tts") || n.contains("qwen3_tts"));
    if !qwen {
        return None;
    }
    let has = |w: &str| names.iter().any(|n| n.contains(w));
    if has("customvoice") {
        Some("CustomVoice")
    } else if has("voicedesign") {
        Some("VoiceDesign")
    } else if has("base") {
        Some("Base")
    } else {
        None
    }
}

/// The voices of a request as one engine call takes them: at most one preset, design or clone
/// for the whole text, and for a script one cloned voice for speaker 0 (Drama 3's one
/// reference). Anything else is refused, never mapped.
struct Voices<'a> {
    preset: Option<&'a str>,
    design: Option<&'a str>,
    clone: Option<(&'a [u8], &'a str, Option<&'a str>)>,
}

fn put<'a>(v: &mut Voices<'a>, voice: &'a Voice) {
    match voice {
        Voice::Preset { name } => v.preset = Some(name.trim()),
        Voice::Design { description } => v.design = Some(description.trim()),
        Voice::Clone { audio, format, text, .. } => {
            v.clone = Some((audio.as_slice(), format.as_str(), text.as_deref().map(str::trim).filter(|t| !t.is_empty())));
        }
        _ => {}
    }
}

fn voices<'a>(model: &str, request: &'a SpeakRequest) -> Result<Voices<'a>, RemoteError> {
    let mut v = Voices { preset: None, design: None, clone: None };
    if let Some(voice) = &request.voice {
        put(&mut v, voice);
    }
    for sv in &request.voices {
        let only_clone = matches!(sv.voice, Voice::Clone { .. }) && sv.speaker == 0 && !matches!(request.voice, Some(Voice::Clone { .. }));
        if !only_clone {
            return Err(bad_request(format!("`{model}` takes a per-speaker voice only as one cloned voice for speaker 0")));
        }
        put(&mut v, &sv.voice);
    }
    Ok(v)
}

/// The `/v1/audio/speech` body for `request` on `engine`, against the model's `profile`
/// (already checked to [cover](SpeechProfile::covers) the request's needs):
///
/// - llama-server: `input`, `voice` (a preset), `response_format` (`wav` or `pcm`),
///   `stream_format: "audio"`, `speed`, `lang`, `seed`, `speaker_ref_b64` and
///   `speaker_ref_text` (a clone), `raw_prompt` (raw input). A designed voice is the
///   description in parentheses before the text (VoxCPM2); a script's turns get speaker tags.
/// - vLLM-Omni: `input`, `voice`, `task_type` (Qwen3-TTS checkpoints), `instructions` (the
///   description for a designed voice, style otherwise), `language`, `speed`, `seed`,
///   `response_format`, `stream` with `stream_format: "audio"`, `ref_audio` as a `data:` URL,
///   `ref_text` and `cache_salt` (a clone).
///
/// A reference transcript the model doesn't use is left out and named in `meta.ignored`. A
/// language or sample rate the profile doesn't list is refused.
pub(crate) fn body(
    engine: EngineKind,
    model: &str,
    request: &SpeakRequest,
    profile: &SpeechProfile,
    task: Option<&str>,
) -> Result<SpeechBody, RemoteError> {
    let format = request.format.or(profile.default_format()).unwrap_or(AudioFormat::Wav);
    let response_format = format_name(engine, model, format)?;
    let stream = request.stream && profile.streaming;
    if let Some(lang) = request.language.as_deref().map(str::trim).filter(|l| !l.is_empty())
        && let Some(list) = &profile.languages
        && !list.iter().any(|l| l.eq_ignore_ascii_case(lang))
    {
        return Err(bad_request(format!("`{model}` does not speak `{lang}` (languages: {})", list.join(", "))));
    }
    if let Some(hz) = request.sample_rate {
        match profile.sample_rate {
            Some(own) if own == hz => {}
            Some(own) => return Err(bad_request(format!("`{model}` speaks at {own} Hz and does not resample; asked for {hz} Hz"))),
            None => return Err(bad_request(format!("`{model}` reports no sample rate, so {hz} Hz can't be promised"))),
        }
    }
    let v = voices(model, request)?;
    let mut meta = Meta::default();
    let uses_text = profile.clone.is_some_and(|m| m.uses_text());
    let clone_text = v.clone.and_then(|(_, _, t)| t);
    if clone_text.is_some() && !uses_text {
        meta = meta.with_ignored("reference_text");
    }
    let (mut text, raw) = match &request.input {
        SpeakInput::Text(t) => (t.clone(), false),
        SpeakInput::Raw(t) => (t.clone(), true),
        other => (script_text(other).unwrap_or_default(), false),
    };
    let mut obj = Map::new();
    obj.insert("model".into(), model.into());
    obj.insert("response_format".into(), response_format.into());
    if let Some(speed) = request.speed {
        obj.insert("speed".into(), json!(f64::from(speed) / 1000.0));
    }
    if let Some(seed) = request.seed {
        obj.insert("seed".into(), seed.into());
    }
    if let Some(name) = v.preset {
        obj.insert("voice".into(), name.into());
    }
    let lang = request.language.as_deref().map(str::trim).filter(|l| !l.is_empty());
    let instructions = request.instructions.as_deref().map(str::trim).filter(|i| !i.is_empty());
    match engine {
        EngineKind::Llamacpp => {
            if let Some(d) = v.design {
                text = format!("({d}){text}");
            }
            if raw {
                obj.insert("raw_prompt".into(), true.into());
            }
            if let Some(l) = lang {
                obj.insert("lang".into(), l.into());
            }
            if let Some((audio, _, _)) = v.clone {
                obj.insert("speaker_ref_b64".into(), base64::engine::general_purpose::STANDARD.encode(audio).into());
                if let Some(t) = clone_text.filter(|_| uses_text) {
                    obj.insert("speaker_ref_text".into(), t.into());
                }
            }
            if instructions.is_some() {
                meta = meta.with_ignored("instructions");
            }
            if stream {
                obj.insert("stream_format".into(), "audio".into());
            }
        }
        EngineKind::VllmOmni => {
            if raw {
                return Err(bad_request(format!("`{model}` runs on vLLM-Omni, which takes no raw input")));
            }
            if let Some(t) = task {
                obj.insert("task_type".into(), t.into());
            }
            let said = match (v.design, instructions) {
                (Some(d), Some(i)) => Some(format!("{d} {i}")),
                (Some(d), None) => Some(d.to_string()),
                (None, i) => i.map(str::to_string),
            };
            if let Some(i) = said {
                obj.insert("instructions".into(), i.into());
            }
            if let Some(l) = lang {
                obj.insert("language".into(), l.into());
            }
            if let Some((audio, format, _)) = v.clone {
                let url = format!("data:{};base64,{}", clip_mime(format)?, base64::engine::general_purpose::STANDARD.encode(audio));
                obj.insert("ref_audio".into(), url.into());
                if let Some(t) = clone_text.filter(|_| uses_text) {
                    obj.insert("ref_text".into(), t.into());
                }
                obj.insert("cache_salt".into(), cache_salt(&request.id).into());
            }
            if stream {
                obj.insert("stream".into(), true.into());
                obj.insert("stream_format".into(), "audio".into());
            }
        }
        other => return Err(bad_request(format!("`{model}` runs on {}, which serves no speech", other.name()))),
    }
    obj.insert("input".into(), text.into());
    let pcm_channels = (engine == EngineKind::Llamacpp && format == AudioFormat::Pcm16).then_some(1);
    Ok(SpeechBody { body: Value::Object(obj), format, stream, meta, pcm_channels })
}

/// The preset voice names in a `/props` `voices` or `/v1/audio/voices` answer: a list of names,
/// or of objects with a `name` (or `id`, `voice`), at the top or under `voices` or `data`.
pub(crate) fn voice_names(v: &Value) -> Option<Vec<String>> {
    let list = v.as_array().or_else(|| v["voices"].as_array()).or_else(|| v["data"].as_array())?;
    let names: Vec<String> = list
        .iter()
        .filter_map(|x| x.as_str().or_else(|| x["name"].as_str()).or_else(|| x["id"].as_str()).or_else(|| x["voice"].as_str()))
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .collect();
    Some(names)
}

/// What llama-server reports for a speech model, over what its API always does (`wav` and
/// `pcm16`, streamed): the preset voices in `/props` `voices`, when it lists them.
pub(crate) fn llama_reported(props: Option<&Value>) -> weirdnet_model::SpeechProfile {
    use weirdnet_model::speech::AudioFormat as F;
    let mut p = weirdnet_model::SpeechProfile::new().with_formats([F::Wav, F::Pcm16]).with_streaming(true);
    if let Some(names) = props.and_then(|v| voice_names(&v["voices"])).filter(|n| !n.is_empty()) {
        p = p.with_presets(names);
    }
    p
}

/// What vLLM-Omni reports, over what its speech API always does (`wav`, `mp3`, `flac`, `pcm16`,
/// `opus`, streamed, with a speed): the preset voices from `GET /v1/audio/voices` and the sample
/// rate its answer to a one-word probe gave.
pub(crate) fn omni_reported(voices: Option<&Value>, sample_rate: Option<u32>) -> weirdnet_model::SpeechProfile {
    use weirdnet_model::speech::AudioFormat as F;
    let mut p = weirdnet_model::SpeechProfile::new().with_formats([F::Wav, F::Mp3, F::Flac, F::Pcm16, F::Opus]).with_streaming(true).with_speed(true);
    if let Some(names) = voices.and_then(voice_names).filter(|n| !n.is_empty()) {
        p = p.with_presets(names);
    }
    if let Some(hz) = sample_rate {
        p = p.with_sample_rate(hz);
    }
    p
}

/// The sample rate an answer's headers give: vLLM-Omni's `x-audio-sample-rate`, else a `rate`
/// parameter on the `Content-Type` (`audio/pcm;rate=24000`).
pub(crate) fn header_rate(headers: &reqwest::header::HeaderMap) -> Option<u32> {
    let parse = |s: &str| s.trim().parse::<u32>().ok().filter(|r| *r > 0);
    if let Some(r) = headers.get("x-audio-sample-rate").and_then(|v| v.to_str().ok()).and_then(parse) {
        return Some(r);
    }
    let ct = headers.get(reqwest::header::CONTENT_TYPE)?.to_str().ok()?;
    ct.split(';').skip(1).find_map(|p| p.trim().strip_prefix("rate=").and_then(parse))
}

/// The channel count a `Content-Type` gives (`audio/pcm;rate=24000;channels=2`).
pub(crate) fn header_channels(headers: &reqwest::header::HeaderMap) -> Option<u16> {
    let ct = headers.get(reqwest::header::CONTENT_TYPE)?.to_str().ok()?;
    ct.split(';').skip(1).find_map(|p| p.trim().strip_prefix("channels=").and_then(|c| c.trim().parse::<u16>().ok()).filter(|c| *c > 0))
}

/// The sample rate and channels in a WAV header (`fmt ` chunk), when `bytes` starts with one.
pub(crate) fn wav_format(bytes: &[u8]) -> Option<(u32, u16)> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().ok()?) as usize;
        let body = at + 8;
        if &bytes[at..at + 4] == b"fmt " {
            if size < 16 || body + 8 > bytes.len() {
                return None;
            }
            let channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().ok()?);
            let rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().ok()?);
            return (rate > 0 && channels > 0).then_some((rate, channels));
        }
        at = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

/// The length in milliseconds of a WAV file whose first bytes are `head` and whose whole length
/// is `total` (for a stream, where the header's sizes may be placeholders).
pub(crate) fn wav_ms(head: &[u8], total: u64) -> Option<u64> {
    if head.len() < 12 || &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12;
    let mut byte_rate = None;
    while at + 8 <= head.len() {
        let size = u32::from_le_bytes(head[at + 4..at + 8].try_into().ok()?) as usize;
        let body = at + 8;
        match &head[at..at + 4] {
            b"fmt " if size >= 12 && body + 12 <= head.len() => {
                byte_rate = Some(u32::from_le_bytes(head[body + 8..body + 12].try_into().ok()?)).filter(|r| *r > 0);
            }
            b"data" => return byte_rate.map(|r| total.saturating_sub(body as u64) * 1000 / u64::from(r)),
            _ => {}
        }
        at = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_agent::remote::infer::v2::ScriptTurn;
    use weirdnet_agent::speech::CloneMode;

    fn kokoro() -> SpeechProfile {
        SpeechProfile::new()
            .with_presets(vec!["af_heart".into(), "am_adam".into()])
            .with_formats(vec![AudioFormat::Wav, AudioFormat::Pcm16])
            .with_streaming(true)
            .with_speed(true)
            .with_raw(true)
            .with_languages(vec!["en-us".into(), "en-gb".into()])
            .with_sample_rate(24_000)
    }

    fn omni() -> SpeechProfile {
        SpeechProfile::new()
            .with_presets(vec!["vivian".into(), "ryan".into()])
            .with_formats(vec![AudioFormat::Wav, AudioFormat::Mp3, AudioFormat::Flac, AudioFormat::Pcm16, AudioFormat::Opus])
            .with_streaming(true)
            .with_speed(true)
            .with_instructions(true)
            .with_sample_rate(24_000)
    }

    fn sent_no_reference_url(v: &Value) {
        let s = v.to_string();
        assert!(!s.contains("http://") && !s.contains("https://") && !s.contains("file://"), "{s}");
    }

    #[test]
    fn llama_server_bodies() {
        let r = SpeakRequest::text("Hello there").with_voice(Voice::preset("am_adam")).with_speed(1200).with_language("en-us").with_seed(7);
        let b = body(EngineKind::Llamacpp, "kokoro", &r, &kokoro(), None).unwrap();
        assert_eq!(
            b.body,
            json!({"model": "kokoro", "input": "Hello there", "voice": "am_adam", "response_format": "wav", "speed": 1.2, "lang": "en-us", "seed": 7})
        );
        assert_eq!((b.format, b.stream), (AudioFormat::Wav, false));

        // pcm16 is `pcm`; streaming asks for raw audio.
        let r = SpeakRequest::text("hi").with_format(AudioFormat::Pcm16).with_stream(true);
        let b = body(EngineKind::Llamacpp, "kokoro", &r, &kokoro(), None).unwrap();
        assert_eq!((b.body["response_format"].as_str(), b.body["stream_format"].as_str(), b.stream), (Some("pcm"), Some("audio"), true));
        assert_eq!(b.pcm_channels, Some(1), "llama-server's pcm is mono");
        // Not streamed when the model doesn't stream.
        let b = body(EngineKind::Llamacpp, "kokoro", &r, &kokoro().with_streaming(false), None).unwrap();
        assert!(!b.stream && b.body.get("stream_format").is_none());

        // Raw phonemes.
        let b = body(EngineKind::Llamacpp, "kokoro", &SpeakRequest::raw("həlˈoʊ"), &kokoro(), None).unwrap();
        assert_eq!((b.body["input"].as_str(), b.body["raw_prompt"].as_bool()), (Some("həlˈoʊ"), Some(true)));

        // VoxCPM2: a designed voice is a prefix; a clone sends the clip, and its transcript is
        // named as ignored rather than sent.
        let vox = SpeechProfile::new().with_design(true).with_clone(CloneMode::Audio).with_formats(vec![AudioFormat::Wav]).with_sample_rate(48_000);
        let b =
            body(EngineKind::Llamacpp, "voxcpm2", &SpeakRequest::text("Good morning.").with_voice(Voice::design("a warm, low voice")), &vox, None)
                .unwrap();
        assert_eq!(b.body["input"], "(a warm, low voice)Good morning.");
        let clone = Voice::clone_from(vec![1, 2, 3], "wav", Some("what it says".into()), true);
        let b = body(EngineKind::Llamacpp, "voxcpm2", &SpeakRequest::text("hi").with_voice(clone.clone()), &vox, None).unwrap();
        assert_eq!(b.body["speaker_ref_b64"], "AQID");
        assert!(b.body.get("speaker_ref_text").is_none());
        assert_eq!(b.meta.ignored, vec!["reference_text".to_string()]);
        sent_no_reference_url(&b.body);
        // A model that uses the transcript gets it.
        let b = body(EngineKind::Llamacpp, "q", &SpeakRequest::text("hi").with_voice(clone), &vox.with_clone(CloneMode::AudioAndText), None).unwrap();
        assert_eq!(b.body["speaker_ref_text"], "what it says");
        assert!(b.meta.ignored.is_empty());

        // Drama 3: speaker tags, the clone is speaker 0's.
        let drama = SpeechProfile::new().with_script_speakers(16).with_clone(CloneMode::AudioTextRequired).with_formats(vec![AudioFormat::Wav]);
        let r = SpeakRequest::script(vec![ScriptTurn::new(0, " Hi! "), ScriptTurn::new(1, "[sighs] Hello.")])
            .with_speaker_voice(0, Voice::clone_from(vec![9], "wav", Some("ref".into()), true));
        let b = body(EngineKind::Llamacpp, "drama", &r, &drama, None).unwrap();
        assert_eq!(b.body["input"], "<|speaker:0|>Hi!\n<|speaker:1|>[sighs] Hello.");
        assert_eq!((b.body["speaker_ref_b64"].as_str(), b.body["speaker_ref_text"].as_str()), (Some("CQ=="), Some("ref")));
        let r = SpeakRequest::script(vec![ScriptTurn::new(0, "a"), ScriptTurn::new(1, "b")]).with_speaker_voice(1, Voice::preset("x"));
        let e = body(EngineKind::Llamacpp, "drama", &r, &drama, None).unwrap_err();
        assert!(e.to_string().contains("speaker 0"), "{e:?}");
    }

    #[test]
    fn llama_server_refusals() {
        let e = body(EngineKind::Llamacpp, "kokoro", &SpeakRequest::text("hi").with_format(AudioFormat::Mp3), &kokoro(), None).unwrap_err();
        assert!(matches!(&e, RemoteError::BadRequest { message } if message.contains("wav and pcm16 only")), "{e:?}");
        let e = body(EngineKind::Llamacpp, "kokoro", &SpeakRequest::text("hi").with_language("fr"), &kokoro(), None).unwrap_err();
        assert!(e.to_string().contains("does not speak `fr`") && e.to_string().contains("en-us"), "{e:?}");
        let e = body(EngineKind::Llamacpp, "kokoro", &SpeakRequest::text("hi").with_sample_rate(16_000), &kokoro(), None).unwrap_err();
        assert!(e.to_string().contains("24000 Hz"), "{e:?}");
        assert!(body(EngineKind::Llamacpp, "kokoro", &SpeakRequest::text("hi").with_sample_rate(24_000), &kokoro(), None).is_ok());
        let e = body(EngineKind::Llamacpp, "k", &SpeakRequest::text("hi").with_sample_rate(24_000), &SpeechProfile::new(), None).unwrap_err();
        assert!(e.to_string().contains("no sample rate"), "{e:?}");
        let e = body(EngineKind::Vllm, "v", &SpeakRequest::text("hi"), &kokoro(), None).unwrap_err();
        assert!(e.to_string().contains("serves no speech"), "{e:?}");
    }

    #[test]
    fn vllm_omni_bodies() {
        let r = SpeakRequest::text("Hello").with_voice(Voice::preset("vivian")).with_instructions("cheerful").with_language("en").with_speed(900);
        let b = body(EngineKind::VllmOmni, "qwen3-tts-1.7b-customvoice", &r, &omni(), Some("CustomVoice")).unwrap();
        assert_eq!(
            b.body,
            json!({"model": "qwen3-tts-1.7b-customvoice", "input": "Hello", "voice": "vivian", "task_type": "CustomVoice",
                "instructions": "cheerful", "language": "en", "speed": 0.9, "response_format": "wav"})
        );
        let b = body(EngineKind::VllmOmni, "q", &SpeakRequest::text("hi").with_format(AudioFormat::Opus).with_stream(true), &omni(), None).unwrap();
        assert_eq!(
            (b.body["response_format"].as_str(), b.body["stream"].as_bool(), b.body["stream_format"].as_str()),
            (Some("opus"), Some(true), Some("audio"))
        );
        assert_eq!(b.format, AudioFormat::Opus);
        // VoiceDesign: the description is the instructions.
        let b = body(
            EngineKind::VllmOmni,
            "d",
            &SpeakRequest::text("hi").with_voice(Voice::design("an old sailor")),
            &omni().with_design(true),
            Some("VoiceDesign"),
        )
        .unwrap();
        assert_eq!((b.body["instructions"].as_str(), b.body["task_type"].as_str()), (Some("an old sailor"), Some("VoiceDesign")));
        // Base: a clone as a data: URL with its transcript and a salt.
        let base = omni().with_clone(CloneMode::AudioAndText);
        let r = SpeakRequest::text("hi").with_voice(Voice::clone_from(vec![1, 2, 3], "MP3", Some("ref words".into()), true));
        let b = body(EngineKind::VllmOmni, "b", &r, &base, Some("Base")).unwrap();
        assert_eq!(b.body["ref_audio"], "data:audio/mpeg;base64,AQID");
        assert_eq!(b.body["ref_text"], "ref words");
        assert_eq!(b.body["cache_salt"].as_str().unwrap(), cache_salt(&r.id));
        assert_ne!(cache_salt("a"), cache_salt("b"));
        sent_no_reference_url(&b.body);
        // A clip format that could be read as a URL or path is refused.
        for bad in ["http://x/a.wav", "../a", "file:a", ""] {
            let r = SpeakRequest::text("hi").with_voice(Voice::clone_from(vec![1], bad, None, true));
            assert!(body(EngineKind::VllmOmni, "b", &r, &base, None).is_err(), "{bad}");
        }
        // Raw input is llama.cpp's.
        let e = body(EngineKind::VllmOmni, "q", &SpeakRequest::raw("x"), &omni().with_raw(true), None).unwrap_err();
        assert!(e.to_string().contains("no raw input"), "{e:?}");
    }

    #[test]
    fn task_types_come_from_the_checkpoint() {
        let l = |id: &str, src: &str| ModelLaunch::new(id, EngineKind::VllmOmni, src);
        assert_eq!(task_type(&l("tts", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")), Some("CustomVoice"));
        assert_eq!(task_type(&l("qwen3-tts-1.7b-voicedesign", "/m/x")), Some("VoiceDesign"));
        assert_eq!(task_type(&l("qwen3-tts-0.6b-base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")), Some("Base"));
        assert_eq!(task_type(&l("voxcpm2", "openbmb/VoxCPM2")), None);
        assert_eq!(task_type(&ModelLaunch::new("qwen3-tts-1.7b-base", EngineKind::Llamacpp, "/m/q.gguf")), None);
    }

    #[test]
    fn reported_profiles_and_headers() {
        let p = llama_reported(Some(&json!({"voices": ["af_heart", {"name": "am_adam"}, ""]})));
        assert_eq!(p.presets, Some(vec!["af_heart".to_string(), "am_adam".to_string()]));
        assert!(p.streaming);
        assert_eq!(llama_reported(Some(&json!({}))).presets, None, "no voices listed: absent");
        let p = omni_reported(Some(&json!({"voices": [{"id": "vivian"}, "ryan"]})), Some(24_000));
        assert_eq!((p.presets.clone(), p.sample_rate, p.speed), (Some(vec!["vivian".into(), "ryan".into()]), Some(24_000), true));
        assert_eq!(omni_reported(Some(&json!([])), None).presets, None);

        let mut h = reqwest::header::HeaderMap::new();
        h.insert("content-type", "audio/pcm;rate=24000".parse().unwrap());
        assert_eq!(header_rate(&h), Some(24_000));
        h.insert("x-audio-sample-rate", "48000".parse().unwrap());
        assert_eq!(header_rate(&h), Some(48_000));
        assert_eq!(header_rate(&reqwest::header::HeaderMap::new()), None);
        assert_eq!(header_channels(&h), None);
        h.insert("content-type", "audio/pcm;rate=44100;channels=2".parse().unwrap());
        assert_eq!(header_channels(&h), Some(2));

        let wav = crate::transcribe::tests::wav(500);
        assert_eq!(wav_format(&wav), Some((16_000, 1)));
        assert_eq!(wav_ms(&wav[..64], wav.len() as u64), Some(500));
        assert_eq!(wav_format(b"OggS...."), None);
    }
}
