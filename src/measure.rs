//! Limits measured when a model loads: what its GGUF files say (`weirdnet-model`'s
//! [`limits::measure`](weirdnet_model::limits::measure)) and the context the engine actually runs
//! it with, per sequence. Nothing here invents a figure: what can't be read stays absent.

use std::path::{Path, PathBuf};

use weirdnet_model::ModelStore;
use weirdnet_model::limits::{LimitSource, LimitsRecord, ModelLimits};

use crate::engine::{EngineKind, ModelLaunch};

/// The last value of one of `names` (`--name VALUE` or `--name=VALUE`) across `lists`, later
/// lists (and later flags) winning, as on an engine's command line.
pub(crate) fn flag(lists: &[&[String]], names: &[&str]) -> Option<String> {
    let mut found = None;
    for args in lists {
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if let Some((k, v)) = a.split_once('=')
                && names.contains(&k)
            {
                found = Some(v.to_string());
            } else if names.contains(&a.as_str())
                && let Some(v) = it.next()
            {
                found = Some(v.clone());
            }
        }
    }
    found
}

/// The projector a llama.cpp launch loads: [`ModelLaunch::mmproj`], else `--mmproj` / `-mm` in
/// the engine's or the model's extra arguments.
pub(crate) fn projector(launch: &ModelLaunch, engine_args: &[String]) -> Option<String> {
    launch.mmproj.clone().or_else(|| flag(&[engine_args, &launch.extra_args], &["--mmproj", "-mm"]))
}

/// llama.cpp `--image-max-tokens`, when the node passes it.
pub(crate) fn image_max_tokens(launch: &ModelLaunch, engine_args: &[String]) -> Option<u32> {
    flag(&[engine_args, &launch.extra_args], &["--image-max-tokens"]).and_then(|v| v.trim().parse().ok())
}

/// What `launch` runs with, as loaded: for a llama.cpp model whose source is a GGUF file, its
/// header and its projector's ([`limits::measure`](weirdnet_model::limits::measure), with
/// `--image-max-tokens` over the header's per-image cap for size-dependent projectors), else
/// only `context` (per sequence, as the engine reported it). `None` when nothing is known.
/// Blocking: reads files.
pub(crate) fn measure(launch: &ModelLaunch, engine_args: &[String], context: Option<u32>) -> Option<LimitsRecord> {
    let from_files = (launch.engine == EngineKind::Llamacpp && Path::new(&launch.source).is_file())
        .then(|| {
            let model = Path::new(&launch.source);
            let proj = if launch.embedding { None } else { projector(launch, engine_args).map(PathBuf::from) };
            match weirdnet_model::limits::measure(model, proj.as_deref(), context) {
                Ok(r) => Some(r),
                Err(e) => {
                    tracing::debug!(model = %launch.id, "limits: {e}");
                    // A projector that doesn't read still leaves the model's own figures.
                    proj.and_then(|_| weirdnet_model::limits::measure(model, None, context).ok())
                }
            }
        })
        .flatten();
    let mut record = match from_files {
        Some(r) => r.with_engine(launch.engine.name()),
        None => {
            let ctx = context?;
            LimitsRecord::new(ModelLimits::new().with_context_tokens(ctx), LimitSource::Measured).with_engine(launch.engine.name()).stamped()
        }
    };
    if let Some(n) = image_max_tokens(launch, engine_args)
        && record.limits.image_patch.is_some()
    {
        record.limits = ModelLimits::new().with_image_tokens(n).or(record.limits);
    }
    Some(record)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// The store's GGUF model whose first file is `source`.
fn stored(store: &ModelStore, source: &str) -> Option<weirdnet_model::StoredModel> {
    let source = Path::new(source);
    store.list().into_iter().find(|m| m.files.first().is_some_and(|f| same_file(&store.dir().join(f), source)))
}

/// Keeps `record` next to `launch`'s files when they are in `store` (a record without file
/// figures is not kept: the context belongs to one run). Blocking.
pub(crate) fn keep(store: &ModelStore, launch: &ModelLaunch, record: &LimitsRecord) {
    if launch.engine != EngineKind::Llamacpp {
        return;
    }
    if let Some(m) = stored(store, &launch.source)
        && let Err(e) = store.record_limits(&m, record)
    {
        tracing::warn!(model = %launch.id, "could not record limits: {e}");
    }
}

/// The record `store` keeps for `launch`'s files, if current and taken with the projector the
/// launch has now, without its context (which belongs to the run that measured it). Blocking.
pub(crate) fn kept(store: &ModelStore, launch: &ModelLaunch, engine_args: &[String]) -> Option<LimitsRecord> {
    if launch.engine != EngineKind::Llamacpp {
        return None;
    }
    let m = stored(store, &launch.source)?;
    let mut r = store.limits(&m)?;
    let now = if launch.embedding { None } else { projector(launch, engine_args) };
    let now = now.as_deref().and_then(|p| Path::new(p).file_name()).map(|n| n.to_string_lossy().into_owned());
    if r.projector != now {
        return None;
    }
    r.limits.context_tokens = None;
    Some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::tests::gguf;
    use serde_json::json;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn flags_are_read_like_a_command_line() {
        let engine = args(&["--image-max-tokens", "512", "--mmproj=/e/p.gguf"]);
        let model = args(&["-mm", "/m/p.gguf", "--image-max-tokens=1024"]);
        assert_eq!(flag(&[&engine, &model], &["--mmproj", "-mm"]).as_deref(), Some("/m/p.gguf"), "the model's own wins");
        assert_eq!(flag(&[&engine], &["--image-max-tokens"]).as_deref(), Some("512"));
        assert_eq!(flag(&[&engine, &model], &["--image-max-tokens"]).as_deref(), Some("1024"));
        assert_eq!(flag(&[&args(&["--image-max-tokens"])], &["--image-max-tokens"]), None, "a flag without a value");
        let mut l = ModelLaunch::new("m", EngineKind::Llamacpp, "/m/m.gguf");
        assert_eq!(projector(&l, &engine).as_deref(), Some("/e/p.gguf"));
        l.mmproj = Some("/own.gguf".into());
        assert_eq!(projector(&l, &engine).as_deref(), Some("/own.gguf"));
    }

    fn model_file(dir: &Path) -> PathBuf {
        let p = dir.join("m.gguf");
        std::fs::write(
            &p,
            gguf(&[
                ("general.architecture", json!("llama")),
                ("llama.block_count", json!(2)),
                ("llama.attention.head_count", json!(4)),
                ("llama.attention.head_count_kv", json!(4)),
                ("llama.embedding_length", json!(256)),
                ("llama.context_length", json!(131072)),
            ]),
        )
        .unwrap();
        p
    }

    fn qwen_projector(dir: &Path) -> PathBuf {
        let p = dir.join("mmproj-q.gguf");
        std::fs::write(
            &p,
            gguf(&[
                ("general.architecture", json!("clip")),
                ("general.type", json!("mmproj")),
                ("clip.has_vision_encoder", json!(true)),
                ("clip.vision.image_size", json!(560)),
                ("clip.vision.patch_size", json!(14)),
                ("clip.projector_type", json!("qwen2.5vl_merger")),
            ]),
        )
        .unwrap();
        p
    }

    #[test]
    fn measures_files_and_context_and_leaves_the_rest_absent() {
        let tmp = tempfile::tempdir().unwrap();
        let model = model_file(tmp.path());
        let proj = qwen_projector(tmp.path());
        let l = ModelLaunch::new("m", EngineKind::Llamacpp, model.to_string_lossy()).with_mmproj(proj.to_string_lossy());
        let r = measure(&l, &[], Some(4096)).unwrap();
        assert_eq!(r.limits.context_tokens, Some(4096), "the context the engine runs, not the trained one");
        assert_eq!(r.trained_context, Some(131072));
        assert_eq!((r.limits.image_patch, r.limits.image_tokens), (Some(28), Some(4096)));
        assert_eq!((r.limits.max_images, r.limits.audio_max_seconds, r.limits.max_request_bytes), (None, None, None), "never invented");
        assert_eq!((r.engine.as_deref(), r.projector.as_deref()), (Some("llamacpp"), Some("mmproj-q.gguf")));

        // `--image-max-tokens` caps a size-dependent projector's cost.
        let mut capped = l.clone();
        capped.extra_args = args(&["--image-max-tokens", "1024"]);
        assert_eq!(measure(&capped, &[], None).unwrap().limits.image_tokens, Some(1024));

        // No context reported: absent, not the trained context.
        assert_eq!(measure(&l, &[], None).unwrap().limits.context_tokens, None);

        // A projector that isn't one: the model's own figures still count.
        let bad = ModelLaunch::new("m", EngineKind::Llamacpp, model.to_string_lossy()).with_mmproj(model.to_string_lossy());
        let r = measure(&bad, &[], Some(2048)).unwrap();
        assert_eq!((r.limits.context_tokens, r.limits.image_tokens, r.projector), (Some(2048), None, None));

        // No file (or not llama.cpp): only the context, when known.
        let missing = ModelLaunch::new("x", EngineKind::Llamacpp, "/nowhere/x.gguf");
        assert_eq!(measure(&missing, &[], Some(512)).unwrap().limits, ModelLimits::new().with_context_tokens(512));
        assert!(measure(&missing, &[], None).is_none());
        let vllm = ModelLaunch::new("v", EngineKind::Vllm, model.to_string_lossy());
        assert_eq!(measure(&vllm, &[], Some(8192)).unwrap().limits, ModelLimits::new().with_context_tokens(8192));
    }

    #[test]
    fn records_are_kept_in_the_store_and_read_back_without_context() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let model = model_file(tmp.path());
        let proj_dir = tempfile::tempdir().unwrap();
        let proj = qwen_projector(proj_dir.path());
        let l = ModelLaunch::new("m", EngineKind::Llamacpp, model.to_string_lossy()).with_mmproj(proj.to_string_lossy());
        assert!(kept(&store, &l, &[]).is_none());
        let r = measure(&l, &[], Some(4096)).unwrap();
        keep(&store, &l, &r);
        let back = kept(&store, &l, &[]).unwrap();
        assert_eq!(back.limits.context_tokens, None, "the context belongs to the run that measured it");
        assert_eq!(back.limits.image_tokens, Some(4096));
        // Another projector now: the record doesn't apply.
        let other = ModelLaunch::new("m", EngineKind::Llamacpp, model.to_string_lossy());
        assert!(kept(&store, &other, &[]).is_none());
        // A model outside the store is not recorded.
        let outside = ModelLaunch::new("o", EngineKind::Llamacpp, proj.to_string_lossy());
        keep(&store, &outside, &r);
        assert!(kept(&store, &outside, &[]).is_none());
    }
}
