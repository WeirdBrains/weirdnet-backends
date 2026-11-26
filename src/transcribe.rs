//! Speech to text on the engines: what a llama.cpp model (Voxtral and other audio-projector
//! models) is asked through `/v1/chat/completions`, and how vLLM's
//! `/v1/audio/transcriptions` answers are read.

use base64::Engine as _;
use serde_json::Value;
use weirdnet_agent::remote::infer::v2::{Request, Sampling, Segment, TranscribeRequest, Transcript, Usage};
use weirdnet_agent::{ContentPart, Message};

/// The instruction a chat model with an audio projector gets with the clip.
pub(crate) fn instruction(request: &TranscribeRequest) -> String {
    let mut s = String::from("Transcribe this audio word for word. Answer with the transcript only, nothing else.");
    if let Some(lang) = request.language.as_deref().map(str::trim).filter(|l| !l.is_empty()) {
        s.push_str(&format!(" The spoken language is {lang}; write the transcript in that language."));
    }
    if let Some(p) = request.prompt.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        s.push_str(&format!(" Names and terms that may occur, or the text just before: {p}"));
    }
    s
}

/// The chat request for a llama.cpp transcription: one user message with the clip as an
/// `input_audio` part (base64) and [`instruction`], greedy sampling.
pub(crate) fn chat_request(request: &TranscribeRequest) -> Request {
    let audio = base64::engine::general_purpose::STANDARD.encode(&request.audio);
    let message = Message::user(vec![ContentPart::audio(audio, request.format.trim().to_ascii_lowercase()), ContentPart::text(instruction(request))]);
    let mut sampling = Sampling::default();
    sampling.temperature = Some(0.0);
    let mut r = Request::messages(vec![message]).with_sampling(sampling);
    r.priority = request.priority;
    r
}

/// The length of a PCM WAV clip in milliseconds, from its header (`fmt ` byte rate and the
/// `data` chunk's size); `None` for anything else or a header that doesn't add up.
pub(crate) fn wav_duration_ms(audio: &[u8]) -> Option<u64> {
    if audio.len() < 12 || &audio[0..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12;
    let mut byte_rate = None;
    while at + 8 <= audio.len() {
        let id = &audio[at..at + 4];
        let size = u32::from_le_bytes(audio[at + 4..at + 8].try_into().ok()?) as usize;
        let body = at + 8;
        match id {
            b"fmt " if size >= 12 && body + 12 <= audio.len() => {
                byte_rate = Some(u32::from_le_bytes(audio[body + 8..body + 12].try_into().ok()?)).filter(|r| *r > 0);
            }
            b"data" => {
                // A streamed WAV may say 0 or 0xFFFFFFFF: then the bytes that are there count.
                let len = if size == 0 || size == u32::MAX as usize { audio.len() - body } else { size.min(audio.len() - body) };
                return byte_rate.map(|r| (len as u64 * 1000) / u64::from(r));
            }
            _ => {}
        }
        at = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

/// A `/v1/audio/transcriptions` answer (`json` or `verbose_json`): the text, and the language,
/// duration (seconds) and segments when given.
pub(crate) fn from_openai(v: &Value) -> Option<Transcript> {
    let text = v["text"].as_str()?;
    let mut t = Transcript::new(text.trim());
    if let Some(l) = v["language"].as_str().filter(|l| !l.is_empty()) {
        t = t.with_language(l);
    }
    if let Some(d) = v["duration"].as_f64().filter(|d| d.is_finite() && *d >= 0.0) {
        t = t.with_duration_ms((d * 1000.0).round() as u64);
    }
    let ms = |x: &Value| x.as_f64().filter(|s| s.is_finite() && *s >= 0.0).map(|s| (s * 1000.0).round() as u64);
    let segments: Vec<Segment> = v["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| Some(Segment::new(ms(&s["start"])?, ms(&s["end"])?, s["text"].as_str()?.trim())))
        .collect();
    Some(t.with_segments(segments))
}

/// Token counts in an OpenAI-style `usage`, when it has them.
pub(crate) fn usage(v: &Value) -> Option<Usage> {
    let p = v["usage"]["prompt_tokens"].as_u64().or(v["usage"]["input_tokens"].as_u64())?;
    let c = v["usage"]["completion_tokens"].as_u64().or(v["usage"]["output_tokens"].as_u64()).unwrap_or(0);
    Some(Usage::new(p, c))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A PCM WAV clip of `millis` of silence at 16 kHz mono 16-bit.
    pub(crate) fn wav(millis: u32) -> Vec<u8> {
        let rate = 16_000u32;
        let data = (rate * 2 * millis / 1000) as usize;
        let mut out = b"RIFF".to_vec();
        out.extend(((36 + data) as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(rate.to_le_bytes());
        out.extend((rate * 2).to_le_bytes());
        out.extend(2u16.to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend((data as u32).to_le_bytes());
        out.extend(std::iter::repeat_n(0u8, data));
        out
    }

    #[test]
    fn wav_lengths() {
        assert_eq!(wav_duration_ms(&wav(1500)), Some(1500));
        assert_eq!(wav_duration_ms(&wav(0)), Some(0));
        let mut list = wav(250);
        // A LIST chunk before data, odd-sized (padded).
        let at = 36;
        let mut extra = b"LIST".to_vec();
        extra.extend(3u32.to_le_bytes());
        extra.extend([1, 2, 3, 0]);
        list.splice(at..at, extra);
        assert_eq!(wav_duration_ms(&list), Some(250));
        assert_eq!(wav_duration_ms(b"ID3\x04 mp3 bytes"), None);
        assert_eq!(wav_duration_ms(&wav(1000)[..20]), None, "truncated before fmt ends");
        let mut streamed = wav(500);
        streamed[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(wav_duration_ms(&streamed), Some(500));
    }

    #[test]
    fn the_chat_request_carries_the_clip_and_the_hints() {
        let r = TranscribeRequest::new(vec![1, 2, 3], " WAV ").with_language("fr").with_prompt("Mjölnir");
        let c = chat_request(&r);
        let msgs = c.inline_messages().unwrap();
        let v = serde_json::to_value(&msgs[0].content).unwrap();
        assert_eq!(v[0], json!({"type": "input_audio", "input_audio": {"data": "AQID", "format": "wav"}}));
        let text = v[1]["text"].as_str().unwrap();
        assert!(text.contains("transcript only") && text.contains("fr") && text.contains("Mjölnir"), "{text}");
        assert_eq!(c.sampling.temperature, Some(0.0));
        let plain = instruction(&TranscribeRequest::new(vec![1], "wav").with_language(" "));
        assert!(!plain.contains("language") && !plain.contains("Names"), "{plain}");
    }

    #[test]
    fn openai_answers_read() {
        let v = json!({"text": " hello world ", "language": "en", "duration": 2.5,
            "segments": [{"start": 0.0, "end": 1.2, "text": " hello"}, {"start": 1.2, "end": 2.5, "text": " world"}, {"start": "x"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 3}});
        let t = from_openai(&v).unwrap();
        assert_eq!(
            t,
            Transcript::new("hello world")
                .with_language("en")
                .with_duration_ms(2500)
                .with_segments(vec![Segment::new(0, 1200, "hello"), Segment::new(1200, 2500, "world")])
        );
        assert_eq!(usage(&v), Some(Usage::new(40, 3)));
        assert_eq!(from_openai(&json!({"text": "hi"})).unwrap(), Transcript::new("hi"));
        assert_eq!(usage(&json!({"usage": {"type": "duration", "seconds": 3}})), None, "no token counts");
        assert!(from_openai(&json!({"error": "x"})).is_none());
    }
}
