//! models/1 uploads: local files sent to a node in chunks, resumable, checked before anything is
//! placed; [`ModelsCommand::Fetch`] so a node can pull an upload or a stored entry from another
//! node ([`PullSource::Peer`]); [`ModelsCommand::Disk`]; and the [`WritePolicy`] for all writes.
//!
//! An upload is one stream: the client sends [`ModelsCommand::Upload`] with an [`Upload`] (the
//! file list with sizes and SHA-256s, and a [`Provenance`]); the node checks names, size, quota,
//! free disk and, for adapters and projectors, the base, then answers
//! [`ModelsReply::UploadReady`] with the bytes it already holds of each file. The client sends
//! [`UploadFrame::Chunk`]s of at most [`CHUNK`] bytes, file by file from there, then
//! [`UploadFrame::Done`]. Each GGUF or safetensors header is checked as soon as enough of it has
//! arrived, so a wrong file is refused early. On `Done` the node checks every file's size,
//! SHA-256 and header, places the entry in the store ([`ModelStore::import_files`]), serves it
//! when asked and answers [`ModelsReply::Uploaded`]. Closing the stream cancels; the received
//! parts stay in the staging directory (outside the store) for a resume until the policy's TTL.

use std::collections::HashSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncSeekExt};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError, from_cbor, to_cbor};
use weirdnet_model::artifact::{self, Artifact, ArtifactFile, EarlyHeader, ORIGIN_UPLOAD, Provenance, ProvenanceKind};
use weirdnet_model::gguf::{self, GgufKind};
use weirdnet_model::hf::split_part;
use weirdnet_model::{Imported, ModelLayout, ModelStore, Progress};

use super::{MODELS_PROTOCOL, ModelsClient, ModelsCommand, ModelsReply, ModelsRequest, ModelsService, PROGRESS_EVERY, PlacedVia, Pull, Pulled};
use crate::engine::{EngineKind, LoraAdapter, ModelLaunch};
use crate::error::bad_request;
use crate::supervisor::AdapterChange;

/// Bytes of file data in one [`UploadFrame::Chunk`] or [`ModelsReply::Chunk`] at most: under
/// the 64 KiB frame limit with room for the framing.
pub const CHUNK: usize = 48 * 1024;
/// The offer a staged upload was made with, in its staging directory.
const OFFER: &str = "offer.json";
/// Received files, under their artifact paths, in a staged upload's directory.
const FILES: &str = "files";

/// Stable reasons for refusing a write: a refusal is `RemoteError::Refused` with the message
/// `reason: details` ([`refusal_reason`] reads the reason back).
pub mod refusal {
    /// The id is in use (send `replace` to replace an uploaded entry).
    pub const EXISTS: &str = "exists";
    /// Larger than [`WritePolicy::max_upload`](super::WritePolicy::max_upload).
    pub const TOO_BIG: &str = "too-big";
    /// Not enough free disk after [`WritePolicy::keep_free`](super::WritePolicy::keep_free).
    pub const NO_SPACE: &str = "no-space";
    /// Over [`WritePolicy::quota`](super::WritePolicy::quota) for uploaded entries.
    pub const QUOTA: &str = "quota";
    /// The node's engine (or the base's) can't serve this format.
    pub const WRONG_ENGINE: &str = "wrong-engine";
    /// An adapter's architecture differs from its base's.
    pub const WRONG_ARCH: &str = "wrong-arch";
    /// The adapter's or projector's base is missing or not served here.
    pub const NO_BASE: &str = "no-base";
    /// A file name or id that isn't plain, or a refused file type (pickles, code).
    pub const BAD_NAME: &str = "bad-name";
    /// A header that can't be read or says something else than the offer.
    pub const BAD_FORMAT: &str = "bad-format";
    /// Offered as an adapter, but the file is not a LoRA adapter.
    pub const NOT_ADAPTER: &str = "not-adapter";
    /// The write policy's authorizer refused the caller.
    pub const NOT_ALLOWED: &str = "not-allowed";
    /// A file's bytes don't match its SHA-256 (the parts are dropped; send it again).
    pub const CHECKSUM: &str = "checksum";
    /// No such upload or stored entry to fetch.
    pub const NOT_FOUND: &str = "not-found";

    /// Every reason.
    pub const ALL: &[&str] =
        &[EXISTS, TOO_BIG, NO_SPACE, QUOTA, WRONG_ENGINE, WRONG_ARCH, NO_BASE, BAD_NAME, BAD_FORMAT, NOT_ADAPTER, NOT_ALLOWED, CHECKSUM, NOT_FOUND];
}

pub(crate) fn refuse(reason: &str, message: impl std::fmt::Display) -> RemoteError {
    RemoteError::refused(format!("{reason}: {message}"))
}

/// The [`refusal`] reason of a refused write, if it is one.
pub fn refusal_reason(e: &RemoteError) -> Option<&'static str> {
    match e {
        RemoteError::Refused { message } => {
            let reason = message.split_once(": ").map(|(r, _)| r).unwrap_or(message);
            refusal::ALL.iter().find(|r| **r == reason).copied()
        }
        _ => None,
    }
}

/// What to upload: the files (paths, sizes and SHA-256s from
/// [`artifact::inspect`]), where they came from, and what to do with them.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Upload {
    /// The id it is stored (and served) as.
    pub id: String,
    pub files: Vec<ArtifactFile>,
    /// Its kind decides what the node expects (model, adapter or projector). The node fills in
    /// `uploaded_by` (the caller's peer id) and `uploaded_ms` when they are missing, and sets
    /// the files and revision from the bytes.
    pub provenance: Provenance,
    /// An adapter's base: a model served on the node.
    #[serde(default)]
    pub adapter_for: Option<String>,
    /// A projector's model: a llama.cpp model served on the node.
    #[serde(default)]
    pub projector_for: Option<String>,
    /// Serve it once placed (a model added to the supervisor, an adapter on its base, a
    /// projector on its model).
    #[serde(default)]
    pub serve: bool,
    /// Replace an uploaded entry with the same id.
    #[serde(default)]
    pub replace: bool,
    /// Store it without serving; skips the engine and base checks.
    #[serde(default)]
    pub store_only: bool,
}

impl Upload {
    /// An upload of `artifact` as `id`; the record's kind follows the artifact.
    pub fn new(id: impl Into<String>, artifact: &Artifact, mut provenance: Provenance) -> Self {
        provenance.kind = artifact.kind.provenance_kind();
        Self { id: id.into(), files: artifact.files.clone(), provenance, ..Default::default() }
    }

    pub fn with_adapter_for(mut self, base: impl Into<String>) -> Self {
        self.adapter_for = Some(base.into());
        self
    }

    pub fn with_projector_for(mut self, model: impl Into<String>) -> Self {
        self.projector_for = Some(model.into());
        self
    }

    pub fn serving(mut self) -> Self {
        self.serve = true;
        self
    }

    pub fn replacing(mut self) -> Self {
        self.replace = true;
        self
    }

    pub fn store_only(mut self) -> Self {
        self.store_only = true;
        self
    }

    /// The staging key: the files' revision digits, so the same files resume whoever sends them.
    pub fn key(&self) -> String {
        artifact::revision(&self.files).trim_start_matches("sha256:").to_string()
    }

    /// Bytes of all files.
    pub fn size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// What the client sends after [`ModelsReply::UploadReady`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UploadFrame {
    /// Bytes of file `file` (an index into [`Upload::files`]) at `offset`, which must be where
    /// the node's copy ends; at most [`CHUNK`] bytes.
    Chunk {
        file: u32,
        offset: u64,
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    /// Every file is sent.
    Done,
}

/// Where a pull gets its files.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PullSource {
    /// The Hugging Face Hub (or the store's mirror): [`Pull::repo`].
    #[default]
    Hub,
    /// Another node's staged upload (its key) or uploaded entry (its id), over models/1
    /// [`ModelsCommand::Fetch`], checked against the same hashes. The pulling node needs a peer
    /// dialer ([`ModelsService::with_peer_dialer`]).
    Peer { node: String, upload: String },
}

/// Free space to keep on the store's filesystem after an upload: the larger of `percent` of
/// the filesystem and `bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct KeepFree {
    pub percent: u8,
    pub bytes: u64,
}

impl Default for KeepFree {
    /// 10 percent or 20 GiB, the larger.
    fn default() -> Self {
        Self { percent: 10, bytes: 20 << 30 }
    }
}

impl KeepFree {
    pub fn new(percent: u8, bytes: u64) -> Self {
        Self { percent: percent.min(100), bytes }
    }

    /// Keep nothing free (tests, scratch nodes).
    pub fn none() -> Self {
        Self { percent: 0, bytes: 0 }
    }

    /// Bytes to keep free on a filesystem of `total` bytes.
    pub fn of(&self, total: u64) -> u64 {
        (total / 100).saturating_mul(self.percent.min(100) as u64).max(self.bytes)
    }
}

/// Rules for writes ([`ModelsCommand::Upload`], [`ModelsCommand::Fetch`] and
/// [`ModelsCommand::Remove`]). The default lets in every caller the service admits (the owner's
/// linked devices, decided by m3talsmith on 2026-11-24), keeps [`KeepFree::default`] free, sets
/// no size limit or quota, and drops abandoned parts after 7 days.
#[derive(Clone)]
#[non_exhaustive]
pub struct WritePolicy {
    /// A second authorizer for writes, after the service's own (for example
    /// `RosterAuth::require_roles` with a write role). `None`: the service's own decides.
    pub auth: Option<Arc<dyn Authorize>>,
    /// Bytes per upload at most.
    pub max_upload: Option<u64>,
    /// Bytes of everything uploaded (provenance origin `upload`) on the node at most.
    pub quota: Option<u64>,
    pub keep_free: KeepFree,
    /// Staged uploads untouched this long are removed.
    pub part_ttl: Duration,
}

impl Default for WritePolicy {
    fn default() -> Self {
        Self { auth: None, max_upload: None, quota: None, keep_free: KeepFree::default(), part_ttl: Duration::from_secs(7 * 24 * 3600) }
    }
}

impl std::fmt::Debug for WritePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WritePolicy")
            .field("auth", &self.auth.as_ref().map(|_| "<authorizer>"))
            .field("max_upload", &self.max_upload)
            .field("quota", &self.quota)
            .field("keep_free", &self.keep_free)
            .field("part_ttl", &self.part_ttl)
            .finish()
    }
}

impl WritePolicy {
    pub fn with_auth(mut self, auth: Arc<dyn Authorize>) -> Self {
        self.auth = Some(auth);
        self
    }

    pub fn with_max_upload(mut self, bytes: u64) -> Self {
        self.max_upload = Some(bytes);
        self
    }

    pub fn with_quota(mut self, bytes: u64) -> Self {
        self.quota = Some(bytes);
        self
    }

    pub fn with_keep_free(mut self, keep: KeepFree) -> Self {
        self.keep_free = keep;
        self
    }

    pub fn with_part_ttl(mut self, ttl: Duration) -> Self {
        self.part_ttl = ttl;
        self
    }
}

/// [`ModelsCommand::Disk`]: the node's room for uploads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiskReport {
    /// Free bytes on the store's filesystem.
    pub free: u64,
    pub total: u64,
    /// Bytes the node keeps free ([`WritePolicy::keep_free`]).
    pub keep_free: u64,
    pub max_upload: Option<u64>,
    pub quota: Option<u64>,
    /// Bytes stored with provenance origin `upload`.
    pub uploaded: u64,
    /// Bytes of staged (unfinished) uploads.
    pub staged: u64,
}

/// An upload being received (or pulled from a peer).
struct Receiving {
    key: String,
    dir: PathBuf,
    upload: Upload,
    have: Vec<u64>,
    next_check: Vec<u64>,
    checked: Vec<bool>,
    /// For a GGUF adapter: the base GGUF's architecture.
    base_arch: Option<String>,
    open: Option<(usize, std::fs::File)>,
    last_progress: Option<Instant>,
}

impl Receiving {
    fn files_dir(&self) -> PathBuf {
        self.dir.join(FILES)
    }

    fn part(&self, i: usize) -> PathBuf {
        self.files_dir().join(format!("{}.part", self.upload.files[i].path))
    }

    fn done_path(&self, i: usize) -> PathBuf {
        self.files_dir().join(&self.upload.files[i].path)
    }

    fn total(&self) -> u64 {
        self.upload.size()
    }

    fn done(&self) -> u64 {
        self.have.iter().sum()
    }

    fn progress_due(&mut self) -> bool {
        let due = self.done() == self.total() || self.last_progress.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY);
        if due {
            self.last_progress = Some(Instant::now());
        }
        due
    }
}

/// Holds an upload key while it is received; released on drop.
struct Claim<'a> {
    set: &'a Mutex<HashSet<String>>,
    key: String,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.set.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.key);
    }
}

/// A plain id: one path part, not hidden.
fn check_id(id: &str) -> Result<(), RemoteError> {
    if id.is_empty() || id.contains('/') || id.contains('\\') || id.starts_with('.') || weirdnet_model::hf::check_file(id).is_err() {
        return Err(refuse(refusal::BAD_NAME, format!("{id:?} is not a valid id")));
    }
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The newest modification time of the files directly in `dir` and `dir/files` (recursively).
fn newest(dir: &Path) -> Option<SystemTime> {
    let mut best = std::fs::metadata(dir).and_then(|m| m.modified()).ok();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let t = if e.file_type().is_ok_and(|t| t.is_dir()) { newest(&e.path()) } else { e.metadata().and_then(|m| m.modified()).ok() };
        best = best.max(t);
    }
    best
}

fn dir_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| if e.file_type().is_ok_and(|t| t.is_dir()) { dir_bytes(&e.path()) } else { e.metadata().map(|m| m.len()).unwrap_or(0) })
        .sum()
}

/// Whether file `path` starts with a header [`artifact::early_header`] reads (a GGUF that isn't
/// a later split part, or a safetensors file).
fn has_header(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    (lower.ends_with(".gguf") && split_part(path).is_none_or(|(_, part, _)| part == 1)) || lower.ends_with(".safetensors")
}

fn gguf_files(u: &Upload) -> bool {
    u.files.iter().any(|f| f.path.to_ascii_lowercase().ends_with(".gguf"))
}

impl ModelsService {
    /// Rules for uploads, fetches and removals (default [`WritePolicy::default`]).
    pub fn with_write_policy(mut self, policy: WritePolicy) -> Self {
        self.write = policy;
        self
    }

    /// Where partial uploads are kept until they finish: outside the store, ideally on the same
    /// filesystem so placing them is a rename. Default: next to the store directory, named after
    /// it with `.uploads` added.
    pub fn with_staging(mut self, dir: impl Into<PathBuf>) -> Self {
        self.staging = dir.into();
        self
    }

    /// Lets this node pull from other nodes ([`PullSource::Peer`]): the dialer and this node's
    /// CBOR `NodeCert` to present to them.
    pub fn with_peer_dialer(mut self, dialer: Arc<dyn Dialer>, cert: Option<Vec<u8>>) -> Self {
        self.peers = Some((dialer, cert));
        self
    }

    pub fn write_policy(&self) -> &WritePolicy {
        &self.write
    }

    pub fn staging_dir(&self) -> &Path {
        &self.staging
    }

    /// The write policy's authorizer, when it has one.
    pub(crate) fn authorize_write(&self, caller: &Caller) -> Result<(), RemoteError> {
        match &self.write.auth {
            Some(a) => a.authorize(caller).map_err(|e| refuse(refusal::NOT_ALLOWED, e)),
            None => Ok(()),
        }
    }

    /// Free space, limits and usage for uploads.
    pub fn disk(&self) -> Result<DiskReport, RemoteError> {
        let space = self.store.disk_space().map_err(super::model_error)?;
        Ok(DiskReport {
            free: space.free,
            total: space.total,
            keep_free: self.write.keep_free.of(space.total),
            max_upload: self.write.max_upload,
            quota: self.write.quota,
            uploaded: self.store.bytes_by_origin(ORIGIN_UPLOAD),
            staged: dir_bytes(&self.staging),
        })
    }

    /// Removes staged uploads untouched for [`WritePolicy::part_ttl`] (except ones being
    /// received). Returns how many were removed.
    pub fn sweep_uploads(&self) -> usize {
        let busy = self.uploading.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut removed = 0;
        for e in std::fs::read_dir(&self.staging).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if busy.contains(&name) || !e.file_type().is_ok_and(|t| t.is_dir()) {
                continue;
            }
            let stale = newest(&e.path()).and_then(|t| t.elapsed().ok()).is_some_and(|age| age >= self.write.part_ttl);
            if stale && std::fs::remove_dir_all(e.path()).is_ok() {
                removed += 1;
            }
        }
        removed
    }

    /// Drops a staged upload's parts (a client's "abandon"). Returns whether there were any.
    pub fn abandon_upload(&self, key: &str) -> Result<bool, RemoteError> {
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(refuse(refusal::BAD_NAME, format!("{key:?} is not an upload key")));
        }
        if self.uploading.lock().unwrap_or_else(|e| e.into_inner()).contains(key) {
            return Err(RemoteError::Busy);
        }
        let dir = self.staging.join(key);
        Ok(dir.exists() && std::fs::remove_dir_all(dir).is_ok())
    }

    fn served_base(&self, base: &str) -> Result<ModelLaunch, RemoteError> {
        self.supervisor
            .launches()
            .into_iter()
            .find(|l| l.id == base)
            .ok_or_else(|| refuse(refusal::NO_BASE, format!("{base} is not served on this node")))
    }

    /// Everything that can be checked before a byte of weights moves.
    fn precheck(&self, u: &Upload) -> Result<(), RemoteError> {
        check_id(&u.id)?;
        if u.files.is_empty() {
            return Err(refuse(refusal::BAD_NAME, "no files"));
        }
        let mut seen = HashSet::new();
        for f in &u.files {
            artifact::check_name(&f.path).map_err(|e| refuse(refusal::BAD_NAME, e))?;
            if !seen.insert(f.path.as_str()) {
                return Err(refuse(refusal::BAD_NAME, format!("{} is listed twice", f.path)));
            }
            if f.sha256.len() != 64 || !f.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(bad_request(format!("{}: the SHA-256 must be 64 hex digits", f.path)));
            }
        }
        let gguf = gguf_files(u);
        let peft = u.files.iter().any(|f| f.path == "adapter_config.json");
        match u.provenance.kind {
            ProvenanceKind::Adapter => {
                if !u.store_only || u.adapter_for.is_some() {
                    let base = u.adapter_for.as_deref().ok_or_else(|| refuse(refusal::NO_BASE, "an adapter needs adapter_for, its base"))?;
                    let launch = self.served_base(base)?;
                    match (launch.engine, gguf, peft) {
                        (EngineKind::Llamacpp, true, _) | (EngineKind::Vllm, false, true) => {}
                        (_, false, false) => return Err(refuse(refusal::NOT_ADAPTER, "neither a LoRA GGUF nor a PEFT adapter directory")),
                        (engine, true, _) => {
                            return Err(refuse(
                                refusal::WRONG_ENGINE,
                                format!("{base} runs on {}, which takes PEFT adapters, not GGUF", engine.name()),
                            ));
                        }
                        (engine, _, _) => {
                            return Err(refuse(
                                refusal::WRONG_ENGINE,
                                format!(
                                    "{base} runs on {}, which takes GGUF adapters, not PEFT; convert it with llama.cpp's convert_lora_to_gguf.py",
                                    engine.name()
                                ),
                            ));
                        }
                    }
                }
                if !u.replace && self.store.adapters().iter().any(|a| a.id == u.id) {
                    return Err(refuse(refusal::EXISTS, format!("adapter {} is stored on this node", u.id)));
                }
            }
            ProvenanceKind::Projector => {
                if !gguf || u.files.len() != 1 {
                    return Err(refuse(refusal::BAD_FORMAT, "a projector is one GGUF file"));
                }
                if !u.store_only || u.projector_for.is_some() {
                    let model = u.projector_for.as_deref().ok_or_else(|| refuse(refusal::NO_BASE, "a projector needs projector_for, its model"))?;
                    let launch = self.served_base(model)?;
                    if launch.engine != EngineKind::Llamacpp {
                        return Err(refuse(refusal::WRONG_ENGINE, format!("{model} runs on {}; projectors are for llama.cpp", launch.engine.name())));
                    }
                }
                if !u.replace && self.store.projectors().iter().any(|p| p.id == u.id) {
                    return Err(refuse(refusal::EXISTS, format!("projector {} is stored on this node", u.id)));
                }
            }
            _ => {
                if !u.store_only {
                    match (self.engine, gguf) {
                        (EngineKind::Llamacpp, true) | (EngineKind::Vllm | EngineKind::Mlx, false) => {}
                        (engine, true) => {
                            return Err(refuse(refusal::WRONG_ENGINE, format!("this node serves {}, which doesn't take GGUF", engine.name())));
                        }
                        (engine, false) => {
                            return Err(refuse(refusal::WRONG_ENGINE, format!("this node serves {}, which takes GGUF only", engine.name())));
                        }
                    }
                }
                let stored = self.store.list().into_iter().chain(self.store.dir_models()).any(|m| m.id == u.id);
                if !u.replace && (stored || (!u.store_only && self.supervisor.launches().iter().any(|l| l.served_ids().contains(&u.id)))) {
                    return Err(refuse(refusal::EXISTS, format!("{} is in use on this node", u.id)));
                }
            }
        }
        let total = u.size();
        if let Some(max) = self.write.max_upload
            && total > max
        {
            return Err(refuse(
                refusal::TOO_BIG,
                format!("{} is larger than this node's {} per upload", weirdnet_model::human(total), weirdnet_model::human(max)),
            ));
        }
        if let Some(quota) = self.write.quota {
            let used = self.store.bytes_by_origin(ORIGIN_UPLOAD);
            if used.saturating_add(total) > quota {
                return Err(refuse(
                    refusal::QUOTA,
                    format!(
                        "{} uploaded plus {} is over this node's {} for uploads",
                        weirdnet_model::human(used),
                        weirdnet_model::human(total),
                        weirdnet_model::human(quota)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Pre-checks, claims the key, sets up staging and reads what is already held.
    fn begin(&self, upload: Upload) -> Result<(Receiving, Claim<'_>, u64), RemoteError> {
        self.precheck(&upload)?;
        self.sweep_uploads();
        let key = upload.key();
        if !self.uploading.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone()) {
            return Err(RemoteError::Busy);
        }
        let claim = Claim { set: &self.uploading, key: key.clone() };
        let dir = self.staging.join(&key);
        let io = |e: std::io::Error| RemoteError::Failed { message: format!("staging: {e}") };
        std::fs::create_dir_all(dir.join(FILES)).map_err(io)?;
        let offer = serde_json::to_vec_pretty(&upload).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        std::fs::write(dir.join(OFFER), offer).map_err(io)?;
        let n = upload.files.len();
        let mut r = Receiving {
            key,
            dir,
            upload,
            have: vec![0; n],
            next_check: vec![0; n],
            checked: vec![false; n],
            base_arch: None,
            open: None,
            last_progress: None,
        };
        for i in 0..n {
            let size = r.upload.files[i].size;
            let done = std::fs::metadata(r.done_path(i)).map(|m| m.len()).ok();
            r.have[i] = match done {
                Some(len) if len == size => size,
                _ => {
                    let _ = std::fs::remove_file(r.done_path(i));
                    match std::fs::metadata(r.part(i)).map(|m| m.len()) {
                        Ok(len) if len <= size => len,
                        _ => {
                            let _ = std::fs::remove_file(r.part(i));
                            0
                        }
                    }
                }
            };
        }
        if r.upload.provenance.kind == ProvenanceKind::Adapter
            && gguf_files(&r.upload)
            && let Some(base) = &r.upload.adapter_for
            && let Ok(launch) = self.served_base(base)
        {
            r.base_arch =
                std::fs::File::open(&launch.source).ok().and_then(|f| gguf::read_header(std::io::BufReader::new(f)).ok()).map(|h| h.architecture);
        }
        let space = ModelStore::new(&self.staging).disk_space().map_err(super::model_error)?;
        let need = r.total() - r.done();
        let keep = self.write.keep_free.of(space.total);
        if space.free < need.saturating_add(keep) {
            return Err(refuse(
                refusal::NO_SPACE,
                format!(
                    "{} to receive, {} free, {} kept free",
                    weirdnet_model::human(need),
                    weirdnet_model::human(space.free),
                    weirdnet_model::human(keep)
                ),
            ));
        }
        for i in 0..n {
            if r.have[i] > 0 {
                self.early_check(&mut r, i)?;
            }
        }
        Ok((r, claim, space.free))
    }

    /// Appends a chunk to file `file`'s part, checks its header when enough has arrived, and
    /// completes the file when it is all there.
    fn write_chunk(&self, r: &mut Receiving, file: u32, offset: u64, data: &[u8]) -> Result<(), RemoteError> {
        let i = file as usize;
        let size = r.upload.files.get(i).map(|f| f.size).ok_or_else(|| RemoteError::protocol(format!("no file {file}")))?;
        if data.len() > CHUNK {
            return Err(RemoteError::protocol(format!("a chunk carries {} bytes, more than {CHUNK}", data.len())));
        }
        if offset != r.have[i] {
            return Err(RemoteError::protocol(format!("{}: chunk at {offset}, expected {}", r.upload.files[i].path, r.have[i])));
        }
        if offset + data.len() as u64 > size {
            return Err(RemoteError::protocol(format!("{}: more than its {size} bytes", r.upload.files[i].path)));
        }
        let io = |e: std::io::Error| RemoteError::Failed { message: format!("staging: {e}") };
        if r.open.as_ref().is_none_or(|(at, _)| *at != i) {
            let part = r.part(i);
            if let Some(d) = part.parent() {
                std::fs::create_dir_all(d).map_err(io)?;
            }
            let f = std::fs::OpenOptions::new().create(true).append(true).open(&part).map_err(io)?;
            r.open = Some((i, f));
        }
        if let Some((_, f)) = r.open.as_mut() {
            f.write_all(data).map_err(io)?;
        }
        r.have[i] += data.len() as u64;
        if r.have[i] == size {
            if let Some((_, f)) = r.open.take() {
                f.sync_all().map_err(io)?;
            }
            std::fs::rename(r.part(i), r.done_path(i)).map_err(io)?;
        }
        if !r.checked[i] && (r.have[i] >= r.next_check[i] || r.have[i] == size) {
            self.early_check(r, i)?;
        }
        Ok(())
    }

    /// Reads file `i`'s header from what has arrived and refuses a wrong file.
    fn early_check(&self, r: &mut Receiving, i: usize) -> Result<(), RemoteError> {
        let f = r.upload.files[i].clone();
        if !has_header(&f.path) {
            r.checked[i] = true;
            return Ok(());
        }
        let have = r.have[i];
        let path = if have == f.size { r.done_path(i) } else { r.part(i) };
        let cap = if f.path.to_ascii_lowercase().ends_with(".gguf") { gguf::HEADER_LIMIT + 1 } else { artifact::SAFETENSORS_HEADER_LIMIT + 8 };
        let mut prefix = Vec::new();
        if let Err(e) = std::fs::File::open(&path).and_then(|file| file.take(have.min(cap)).read_to_end(&mut prefix)) {
            return Err(RemoteError::Failed { message: format!("staging: {e}") });
        }
        let header = match artifact::early_header(&f.path, &prefix, f.size) {
            Ok(None) => {
                r.next_check[i] = have.saturating_mul(2).max(have + (256 << 10));
                return Ok(());
            }
            Ok(Some(h)) => h,
            Err(e) => {
                self.drop_staged(r);
                return Err(refuse(refusal::BAD_FORMAT, e));
            }
        };
        r.checked[i] = true;
        let kind = r.upload.provenance.kind;
        let wrong = match &header {
            EarlyHeader::Gguf(g) => match (kind, &g.kind) {
                (ProvenanceKind::Model, GgufKind::Model) | (ProvenanceKind::Projector, GgufKind::Projector) => None,
                (ProvenanceKind::Adapter, k) if k.is_lora() => match &r.base_arch {
                    Some(arch) if *arch != g.architecture => {
                        Some(refuse(refusal::WRONG_ARCH, format!("{} is for {} models; the base is {arch}", f.path, g.architecture)))
                    }
                    _ => None,
                },
                (ProvenanceKind::Adapter, other) => Some(refuse(refusal::NOT_ADAPTER, format!("{} is a {other:?}, not a LoRA adapter", f.path))),
                (want, other) => Some(refuse(refusal::BAD_FORMAT, format!("{} is a {other:?}, offered as a {want:?}", f.path))),
            },
            EarlyHeader::Safetensors(s) => {
                let mlx = s.metadata.get("format").is_some_and(|v| v == "mlx");
                match (kind, r.upload.store_only, self.engine, mlx) {
                    (ProvenanceKind::Model, false, EngineKind::Vllm, true) => {
                        Some(refuse(refusal::WRONG_ENGINE, format!("{} is MLX; this node serves vllm", f.path)))
                    }
                    (ProvenanceKind::Model, false, EngineKind::Mlx, false) => {
                        Some(refuse(refusal::WRONG_ENGINE, format!("{} is not MLX; this node serves mlx", f.path)))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        match wrong {
            Some(e) => {
                self.drop_staged(r);
                Err(e)
            }
            None => Ok(()),
        }
    }

    fn drop_staged(&self, r: &mut Receiving) {
        r.open = None;
        let _ = std::fs::remove_dir_all(&r.dir);
        r.have.iter_mut().for_each(|h| *h = 0);
    }

    /// Checks and places a fully received upload, then serves it if asked.
    async fn finish(&self, mut r: Receiving, via: PlacedVia, progress: impl FnMut(Progress) -> bool + Send + 'static) -> Result<Pulled, RemoteError> {
        r.open = None;
        for (i, f) in r.upload.files.iter().enumerate() {
            if r.have[i] != f.size {
                return Err(bad_request(format!("{} has {} of {} bytes; send the rest", f.path, r.have[i], f.size)));
            }
        }
        let store = self.store.clone();
        let staged = r.files_dir();
        let (files, provenance, id, replace) = (r.upload.files.clone(), r.upload.provenance.clone(), r.upload.id.clone(), r.upload.replace);
        let mut progress = progress;
        let placed = tokio::task::spawn_blocking(move || store.import_files(&staged, &files, &provenance, &id, replace, &mut progress))
            .await
            .map_err(|e| RemoteError::Failed { message: format!("worker: {e}") })?;
        let imported = match placed {
            Ok(i) => i,
            Err(weirdnet_model::Error::Checksum { name, got }) => {
                self.drop_staged(&mut r);
                return Err(refuse(refusal::CHECKSUM, format!("{name} hashed to {got}; its parts were dropped, send it again")));
            }
            Err(weirdnet_model::Error::Cancelled { .. }) => return Err(RemoteError::Cancelled),
            Err(e) if e.to_string().contains("exists") => return Err(refuse(refusal::EXISTS, e)),
            Err(e @ weirdnet_model::Error::Io(_)) => return Err(RemoteError::Failed { message: e.to_string() }),
            Err(e) => {
                self.drop_staged(&mut r);
                return Err(refuse(refusal::BAD_FORMAT, e));
            }
        };
        let _ = std::fs::remove_dir_all(&r.dir);
        let u = &r.upload;
        let revision = imported.provenance().map(|p| p.revision.clone());
        let serving = if u.serve && !u.store_only { self.serve_imported(u, &imported).await } else { Ok((false, None)) };
        let (served, change) = serving.as_ref().map(|x| *x).unwrap_or((false, None));
        let repo = match &via {
            PlacedVia::Peer { node } => node.clone(),
            _ => String::new(),
        };
        let pulled = Pulled {
            id: imported.id().to_string(),
            repo,
            files: self.store_files(&imported),
            size: imported.size(),
            served,
            loaded: false,
            adapter_for: u.adapter_for.clone().filter(|_| matches!(imported, Imported::Adapter(_))),
            change,
            revision,
            projector_for: u.projector_for.clone().filter(|_| matches!(imported, Imported::Projector(_))),
        };
        self.fire_placed(&pulled, via, Some(&imported), serving.as_ref().err());
        serving.map(|_| pulled)
    }

    /// Hands an imported entry to the supervisor: whether it is served, and the adapter change.
    async fn serve_imported(&self, u: &Upload, imported: &Imported) -> Result<(bool, Option<AdapterChange>), RemoteError> {
        let (mut served, mut change) = (false, None);
        match &imported {
            Imported::Model(m) => {
                let (engine, source) = match m.layout {
                    ModelLayout::Safetensors => {
                        (EngineKind::Vllm, self.store.model_dir(ModelLayout::Safetensors, &m.id).map_err(super::model_error)?)
                    }
                    ModelLayout::Mlx => (EngineKind::Mlx, self.store.model_dir(ModelLayout::Mlx, &m.id).map_err(super::model_error)?),
                    _ => (EngineKind::Llamacpp, self.store.dir().join(&m.files[0])),
                };
                // A format this node's engine doesn't take stays stored, not served.
                if engine == self.engine {
                    if u.replace && self.supervisor.launches().iter().any(|l| l.id == m.id) {
                        self.supervisor.remove_model(&m.id).await?;
                    }
                    self.supervisor.add_model(ModelLaunch::new(m.id.clone(), engine, source.display().to_string()))?;
                    served = true;
                }
            }
            Imported::Adapter(a) => {
                let base = u.adapter_for.clone().ok_or_else(|| refuse(refusal::NO_BASE, "an adapter needs adapter_for"))?;
                if u.replace && self.supervisor.launches().iter().any(|l| l.adapters.iter().any(|x| x.id == a.id)) {
                    self.supervisor.remove_adapter_live(&base, &a.id).await?;
                }
                change = Some(self.supervisor.add_adapter_live(&base, LoraAdapter::new(a.id.clone(), a.path.display().to_string())).await?);
                served = true;
            }
            Imported::Projector(p) => {
                let model = u.projector_for.clone().ok_or_else(|| refuse(refusal::NO_BASE, "a projector needs projector_for"))?;
                change = Some(self.supervisor.set_projector_live(&model, Some(p.path.display().to_string())).await?);
                served = true;
            }
            _ => {}
        }
        Ok((served, change))
    }

    /// An imported entry's files, relative to the store.
    fn store_files(&self, imported: &Imported) -> Vec<String> {
        let under = |dir: &str, id: &str| {
            imported.provenance().map(|p| p.files.iter().map(|f| format!("{dir}/{id}/{}", f.path)).collect()).unwrap_or_default()
        };
        match imported {
            Imported::Model(m) => m.files.clone(),
            Imported::Adapter(a) => under(weirdnet_model::store::ADAPTERS_DIR, &a.id),
            Imported::Projector(p) => under(weirdnet_model::store::PROJECTORS_DIR, &p.id),
            _ => Vec::new(),
        }
    }

    /// Serves one upload stream after the main authorization.
    pub(super) async fn serve_upload(&self, ch: &mut Channel, caller: &Caller, mut upload: Upload) -> Result<(), RemoteError> {
        if let Err(e) = self.authorize_write(caller) {
            return ch.send_cbor(&ModelsReply::Failed(e)).await;
        }
        // The node records who sent it; the client cannot say.
        upload.provenance.uploaded_by = Some(caller.peer.clone());
        upload.provenance.uploaded_ms.get_or_insert_with(now_ms);
        if upload.provenance.origin.is_empty() {
            upload.provenance.origin = ORIGIN_UPLOAD.into();
        }
        let (mut r, _claim, free) = match self.begin(upload) {
            Ok(x) => x,
            Err(e) => return ch.send_cbor(&ModelsReply::Failed(e)).await,
        };
        ch.send_cbor(&ModelsReply::UploadReady { upload: r.key.clone(), have: r.have.clone(), free }).await?;
        loop {
            let frame = match ch.recv_cbor::<UploadFrame>().await {
                Ok(Some(f)) => f,
                // Closed: cancelled. The parts stay for a resume.
                Ok(None) | Err(_) => return Ok(()),
            };
            match frame {
                UploadFrame::Chunk { file, offset, data } => {
                    if let Err(e) = self.write_chunk(&mut r, file, offset, &data) {
                        return ch.send_cbor(&ModelsReply::Failed(e)).await;
                    }
                    if r.progress_due() {
                        ch.send_cbor(&ModelsReply::Progress { done: r.done(), total: r.total() }).await?;
                    }
                }
                UploadFrame::Done => break,
            }
        }
        self.drive(ch, |report| self.finish(r, PlacedVia::Upload, report), ModelsReply::Uploaded).await
    }

    /// Runs `work` (given a progress callback), forwarding progress at most every
    /// [`PROGRESS_EVERY`] and cancelling when the stream closes, then sends its answer.
    async fn drive<'a, F, Fut>(&'a self, ch: &mut Channel, work: F, wrap: fn(Pulled) -> ModelsReply) -> Result<(), RemoteError>
    where
        F: FnOnce(Box<dyn FnMut(Progress) -> bool + Send + 'static>) -> Fut,
        Fut: std::future::Future<Output = Result<Pulled, RemoteError>> + 'a,
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Progress>();
        let stop = cancel.clone();
        let mut last: Option<Instant> = None;
        let report = Box::new(move |p: Progress| {
            if p.done == p.total || last.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) {
                last = Some(Instant::now());
                let _ = tx.send(p);
            }
            !stop.load(Ordering::SeqCst)
        });
        let Channel { tx: out, rx: inbound } = ch;
        let work = work(report);
        tokio::pin!(work);
        let mut closed = false;
        let result = loop {
            tokio::select! {
                r = &mut work => break r,
                Some(p) = rx.recv() => {
                    if !closed {
                        out.send(to_cbor(&ModelsReply::Progress { done: p.done, total: p.total })?).await.ok();
                    }
                }
                r = inbound.recv(), if !closed => {
                    if matches!(r, Ok(None) | Err(_)) {
                        closed = true;
                        cancel.store(true, Ordering::SeqCst);
                    }
                }
            }
        };
        if closed {
            return Ok(());
        }
        while let Ok(p) = rx.try_recv() {
            out.send(to_cbor(&ModelsReply::Progress { done: p.done, total: p.total })?).await?;
        }
        let answer = result.map(wrap).unwrap_or_else(ModelsReply::Failed);
        out.send(to_cbor(&answer)?).await
    }

    /// What `upload_or_id` names for a fetch: a finished staged upload (by key) or an uploaded
    /// entry (by id, with its provenance): its id, files, record and directory.
    fn resolve_fetch(&self, upload_or_id: &str) -> Result<(String, Vec<ArtifactFile>, Option<Provenance>, PathBuf), RemoteError> {
        check_id(upload_or_id)?;
        let staged = self.staging.join(upload_or_id);
        if let Ok(bytes) = std::fs::read(staged.join(OFFER)) {
            let upload: Upload = serde_json::from_slice(&bytes).map_err(|e| RemoteError::Failed { message: format!("staging: {e}") })?;
            let dir = staged.join(FILES);
            let complete = upload.files.iter().all(|f| std::fs::metadata(dir.join(&f.path)).is_ok_and(|m| m.len() == f.size));
            if !complete {
                return Err(refuse(refusal::NOT_FOUND, format!("upload {upload_or_id} is not complete yet")));
            }
            return Ok((upload.id, upload.files, Some(upload.provenance), dir));
        }
        let store = &self.store;
        let model = store.list().into_iter().chain(store.dir_models()).find(|m| m.id == upload_or_id && m.provenance.is_some());
        if let Some(m) = model {
            let dir = match m.layout {
                ModelLayout::Gguf => store.dir().join(&m.files[0]).parent().map(Path::to_path_buf).unwrap_or_else(|| store.dir().to_path_buf()),
                layout => store.model_dir(layout, &m.id).map_err(super::model_error)?,
            };
            let p = m.provenance.expect("filtered");
            return Ok((m.id, p.files.clone(), Some(p), dir));
        }
        if let Some(a) = store.adapters().into_iter().find(|a| a.id == upload_or_id && a.provenance.is_some()) {
            let p = a.provenance.expect("filtered");
            return Ok((a.id.clone(), p.files.clone(), Some(p), store.adapter_dir(&a.id).map_err(super::model_error)?));
        }
        if let Some(pr) = store.projectors().into_iter().find(|p| p.id == upload_or_id && p.provenance.is_some()) {
            let p = pr.provenance.expect("filtered");
            return Ok((pr.id.clone(), p.files.clone(), Some(p), store.projector_dir(&pr.id).map_err(super::model_error)?));
        }
        Err(refuse(refusal::NOT_FOUND, format!("no finished upload or uploaded entry {upload_or_id} on this node")))
    }

    /// Serves one fetch stream: the list, then the bytes from `file` at `offset` to the end.
    pub(super) async fn serve_fetch(&self, ch: &mut Channel, caller: &Caller, upload_or_id: &str, file: u32, offset: u64) -> Result<(), RemoteError> {
        if let Err(e) = self.authorize_write(caller) {
            return ch.send_cbor(&ModelsReply::Failed(e)).await;
        }
        let (id, files, provenance, dir) = match self.resolve_fetch(upload_or_id) {
            Ok(x) => x,
            Err(e) => return ch.send_cbor(&ModelsReply::Failed(e)).await,
        };
        ch.send_cbor(&ModelsReply::FetchReady { id, files: files.clone(), provenance: provenance.map(Box::new) }).await?;
        let io = |e: std::io::Error| RemoteError::Failed { message: e.to_string() };
        for (i, f) in files.iter().enumerate().skip(file as usize) {
            let mut at = if i == file as usize { offset } else { 0 };
            if at > f.size {
                return ch.send_cbor(&ModelsReply::Failed(bad_request(format!("{}: offset {at} is past its end", f.path)))).await;
            }
            let mut src = tokio::fs::File::open(dir.join(&f.path)).await.map_err(io)?;
            src.seek(std::io::SeekFrom::Start(at)).await.map_err(io)?;
            let mut buf = vec![0; CHUNK];
            while at < f.size {
                let want = CHUNK.min((f.size - at) as usize);
                src.read_exact(&mut buf[..want]).await.map_err(io)?;
                ch.send_cbor(&ModelsReply::Chunk { file: i as u32, offset: at, data: buf[..want].to_vec() }).await?;
                at += want as u64;
            }
        }
        ch.send_cbor(&ModelsReply::FetchDone).await
    }

    /// Pulls a staged upload or uploaded entry from another node ([`PullSource::Peer`]),
    /// resuming what is staged here, then places and serves it like an upload.
    pub(super) async fn pull_peer(
        &self,
        pull: &Pull,
        node: &str,
        upload_or_id: &str,
        mut progress: impl FnMut(Progress) -> bool + Send + 'static,
    ) -> Result<Pulled, RemoteError> {
        let (dialer, cert) = self.peers.clone().ok_or_else(|| RemoteError::unavailable("this node has no peer dialer to pull from other nodes"))?;
        let fetch = |file: u32, offset: u64| ModelsRequest {
            cert: cert.clone(),
            command: ModelsCommand::Fetch { upload_or_id: upload_or_id.to_string(), file, offset },
        };
        let ready = |reply: Option<ModelsReply>| match reply {
            Some(ModelsReply::FetchReady { id, files, provenance }) => Ok((id, files, provenance.map(|p| *p))),
            Some(ModelsReply::Failed(e)) => Err(e),
            Some(other) => Err(ModelsClient::unexpected("fetch ready", other)),
            None => Err(RemoteError::Closed),
        };
        let mut ch = dialer.open(node, MODELS_PROTOCOL).await?;
        ch.send_cbor(&fetch(u32::MAX, 0)).await?;
        let (src_id, files, provenance) = ready(ch.recv_cbor().await?)?;
        drop(ch);
        let serve = pull.serve || pull.load;
        let upload = Upload {
            id: pull.model.clone().unwrap_or(src_id),
            files: files.clone(),
            provenance: provenance.unwrap_or_default(),
            adapter_for: pull.adapter_for.clone(),
            projector_for: pull.projector_for.clone(),
            serve,
            replace: false,
            store_only: !serve,
        };
        let (mut r, _claim, _) = self.begin(upload)?;
        if let Some(start) = r.have.iter().zip(&files).position(|(h, f)| *h < f.size) {
            let mut ch = dialer.open(node, MODELS_PROTOCOL).await?;
            ch.send_cbor(&fetch(start as u32, r.have[start])).await?;
            let (_, again, _) = ready(ch.recv_cbor().await?)?;
            if again != files {
                return Err(RemoteError::protocol(format!("{node} changed the file list of {upload_or_id}")));
            }
            loop {
                match ch.recv_cbor::<ModelsReply>().await? {
                    Some(ModelsReply::Chunk { file, offset, data }) => {
                        self.write_chunk(&mut r, file, offset, &data)?;
                        if !progress(Progress { done: r.done(), total: r.total() }) {
                            return Err(RemoteError::Cancelled);
                        }
                    }
                    Some(ModelsReply::FetchDone) => break,
                    Some(ModelsReply::Failed(e)) => return Err(e),
                    Some(other) => return Err(ModelsClient::unexpected("a chunk", other)),
                    None => return Err(RemoteError::Closed),
                }
            }
        }
        let pulled = self.finish(r, PlacedVia::Peer { node: node.to_string() }, progress).await?;
        if pull.load && pulled.served {
            self.supervisor.load(&pulled.id).await?;
            return Ok(Pulled { loaded: true, ..pulled });
        }
        Ok(pulled)
    }
}

/// The local file for artifact path `rel`: under the one directory given, or the given file
/// with that name.
fn local_file(local: &[PathBuf], rel: &str) -> Result<PathBuf, RemoteError> {
    if let [dir] = local
        && dir.is_dir()
    {
        return Ok(dir.join(rel));
    }
    local
        .iter()
        .find(|p| p.file_name().is_some_and(|n| n.to_string_lossy() == rel))
        .cloned()
        .ok_or_else(|| bad_request(format!("{rel} is not among the local files")))
}

impl ModelsClient {
    /// Uploads local files to `node` (resuming what it already holds) and returns what was
    /// placed. `local` is what was given to [`artifact::inspect`]: one directory, or the files.
    /// Dropping the future cancels; the node keeps the parts for a resume.
    pub async fn upload(
        &self,
        node: &str,
        upload: Upload,
        local: &[PathBuf],
        mut progress: impl FnMut(Progress) + Send,
    ) -> Result<Pulled, RemoteError> {
        let paths: Vec<PathBuf> = upload.files.iter().map(|f| local_file(local, &f.path)).collect::<Result<_, _>>()?;
        let files = upload.files.clone();
        let total: u64 = files.iter().map(|f| f.size).sum();
        let mut ch = self.open(node, ModelsCommand::Upload(Box::new(upload))).await?;
        let have = match ch.recv_cbor::<ModelsReply>().await? {
            Some(ModelsReply::UploadReady { have, .. }) if have.len() == files.len() => have,
            Some(ModelsReply::Failed(e)) => return Err(e),
            Some(other) => return Err(Self::unexpected("upload ready", other)),
            None => return Err(RemoteError::Closed),
        };
        let mut done: u64 = have.iter().sum();
        progress(Progress { done, total });
        let io = |e: std::io::Error| bad_request(format!("reading the local files: {e}"));
        let mut buf = vec![0; CHUNK];
        for (i, f) in files.iter().enumerate() {
            let mut at = have[i].min(f.size);
            if at == f.size {
                continue;
            }
            let mut src = tokio::fs::File::open(&paths[i]).await.map_err(io)?;
            src.seek(std::io::SeekFrom::Start(at)).await.map_err(io)?;
            while at < f.size {
                let want = CHUNK.min((f.size - at) as usize);
                src.read_exact(&mut buf[..want]).await.map_err(io)?;
                let frame = to_cbor(&UploadFrame::Chunk { file: i as u32, offset: at, data: buf[..want].to_vec() })?;
                if let Err(e) = ch.tx.send(frame).await {
                    // The node may have refused and closed: its reason beats the send error.
                    return Err(match ch.recv_cbor::<ModelsReply>().await {
                        Ok(Some(ModelsReply::Failed(r))) => r,
                        _ => e,
                    });
                }
                at += want as u64;
                done += want as u64;
                // Answers that arrived meanwhile: progress, or an early refusal.
                while let Some(reply) = ch.rx.recv().now_or_never() {
                    match reply? {
                        Some(bytes) => match from_cbor::<ModelsReply>(&bytes)? {
                            ModelsReply::Progress { done, total } => progress(Progress { done, total }),
                            ModelsReply::Failed(e) => return Err(e),
                            other => return Err(Self::unexpected("progress", other)),
                        },
                        None => return Err(RemoteError::Closed),
                    }
                }
            }
        }
        let _ = done;
        ch.send_cbor(&UploadFrame::Done).await?;
        loop {
            match ch.recv_cbor::<ModelsReply>().await? {
                Some(ModelsReply::Progress { done, total }) => progress(Progress { done, total }),
                Some(ModelsReply::Uploaded(p)) => return Ok(p),
                Some(ModelsReply::Failed(e)) => return Err(e),
                Some(other) => return Err(Self::unexpected("uploaded", other)),
                None => return Err(RemoteError::Closed),
            }
        }
    }

    /// Has every node in `targets` pull uploaded entry `upload.id` from node `stage` (where it
    /// was uploaded), node to node, all at once; adapters and projectors go on the bases the
    /// upload names, and they serve when it does. `progress` gets each target's progress.
    /// Returns each target's result.
    pub async fn replicate(
        &self,
        stage: &str,
        targets: &[String],
        upload: &Upload,
        progress: impl Fn(&str, Progress) + Send + Sync,
    ) -> Vec<(String, Result<Pulled, RemoteError>)> {
        let progress = &progress;
        let runs = targets.iter().map(|t| async move {
            let mut pull = Pull::from_peer(stage, &upload.id);
            pull.adapter_for = upload.adapter_for.clone();
            pull.projector_for = upload.projector_for.clone();
            pull.serve = upload.serve && !upload.store_only;
            let r = self.pull(t, pull, |p| progress(t, p)).await;
            (t.clone(), r)
        });
        futures::future::join_all(runs).await
    }

    /// The node's free space, limits and usage for uploads.
    pub async fn disk(&self, node: &str) -> Result<DiskReport, RemoteError> {
        match self.call(node, ModelsCommand::Disk).await? {
            ModelsReply::Disk(d) => Ok(d),
            other => Err(Self::unexpected("disk", other)),
        }
    }

    /// Drops a staged upload's parts on `node`; returns whether there were any.
    pub async fn abandon_upload(&self, node: &str, key: &str) -> Result<bool, RemoteError> {
        match self.call(node, ModelsCommand::AbandonUpload { upload: key.into() }).await? {
            ModelsReply::Abandoned { removed, .. } => Ok(removed),
            other => Err(Self::unexpected("abandoned", other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use weirdnet_agent::remote::MemoryDialer;
    use weirdnet_model::artifact::{ORIGIN_TRAIN, inspect};

    use super::*;
    use crate::engine::tests::{base_gguf, gguf, lora_gguf};
    use crate::models::Placed;
    use crate::supervisor::{AdapterChange, Supervisor, SupervisorOptions};

    /// A llama model GGUF padded to `len` bytes (several chunks).
    fn model_gguf(len: usize) -> Vec<u8> {
        let mut out = gguf(&[
            ("general.architecture", "llama".into()),
            ("llama.block_count", 1.into()),
            ("llama.attention.head_count", 1.into()),
            ("llama.attention.head_count_kv", 1.into()),
            ("llama.attention.key_length", 1.into()),
        ]);
        out.extend((0..len.saturating_sub(out.len())).map(|i| (i % 251) as u8));
        out
    }

    struct Node {
        svc: Arc<ModelsService>,
        sup: Supervisor,
        _tmp: tempfile::TempDir,
        root: PathBuf,
    }

    fn node(engine: EngineKind, policy: WritePolicy) -> Node {
        node_with(engine, policy, |s| s)
    }

    fn node_with(engine: EngineKind, policy: WritePolicy, more: impl FnOnce(ModelsService) -> ModelsService) -> Node {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|c: &Caller| if c.peer == "nobody" { Err(RemoteError::refused("not linked")) } else { Ok(()) });
        let svc = ModelsService::new(ModelStore::new(root.join("models")), sup.clone(), auth)
            .with_engine(engine)
            .with_staging(root.join("staging"))
            .with_write_policy(policy.with_keep_free(KeepFree::none()));
        Node { svc: Arc::new(more(svc)), sup, _tmp: tmp, root }
    }

    fn route(d: &MemoryDialer, name: &str, svc: Arc<ModelsService>, peer: &'static str) {
        d.route_only(name, &[MODELS_PROTOCOL], move |ch, _| {
            let svc = svc.clone();
            tokio::spawn(async move {
                let _ = svc.serve(ch, Caller::new(peer)).await;
            });
        });
    }

    fn client(n: &Node, peer: &'static str) -> ModelsClient {
        let d = MemoryDialer::new();
        route(&d, "node", n.svc.clone(), peer);
        ModelsClient::new(Arc::new(d))
    }

    fn local(dir: &Path, name: &str, body: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, body).unwrap();
        p
    }

    fn reason(r: Result<Pulled, RemoteError>) -> &'static str {
        let e = r.unwrap_err();
        refusal_reason(&e).unwrap_or_else(|| panic!("not a refusal: {e:?}"))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_model_uploads_resumes_and_serves_with_provenance() {
        let n = node(EngineKind::Llamacpp, WritePolicy::default());
        let src = tempfile::tempdir().unwrap();
        let file = local(src.path(), "tune-Q4_0.gguf", &model_gguf(3 * CHUNK + 1000));
        let found = inspect(std::slice::from_ref(&file)).unwrap();
        let upload = Upload::new("support-r3", &found, Provenance::new(ORIGIN_UPLOAD).with_license("apache-2.0")).serving();

        // A first try that stops after one chunk: the part stays.
        let d = MemoryDialer::new();
        route(&d, "node", n.svc.clone(), "laptop");
        let mut ch = d.open("node", MODELS_PROTOCOL).await.unwrap();
        ch.send_cbor(&ModelsRequest { cert: None, command: ModelsCommand::Upload(Box::new(upload.clone())) }).await.unwrap();
        let Some(ModelsReply::UploadReady { upload: key, have, .. }) = ch.recv_cbor().await.unwrap() else { panic!() };
        assert_eq!((key.clone(), have), (upload.key(), vec![0]));
        let body = std::fs::read(&file).unwrap();
        ch.send_cbor(&UploadFrame::Chunk { file: 0, offset: 0, data: body[..CHUNK].to_vec() }).await.unwrap();
        drop(ch);
        let part = n.root.join("staging").join(&key).join("files/tune-Q4_0.gguf.part");
        for _ in 0..100 {
            if std::fs::metadata(&part).is_ok_and(|m| m.len() == CHUNK as u64) && n.svc.uploading.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(std::fs::metadata(&part).unwrap().len(), CHUNK as u64);

        let mut seen = Vec::new();
        let pulled = client(&n, "laptop").upload("node", upload, &[file], |p| seen.push(p)).await.unwrap();
        assert_eq!(seen[0].done, CHUNK as u64, "resumed where the node's copy ended");
        assert_eq!((pulled.id.as_str(), pulled.served, pulled.revision.as_deref()), ("support-r3", true, Some(found.revision.as_str())));
        assert_eq!(pulled.files, ["support-r3/tune-Q4_0.gguf"]);
        let launch = n.sup.launches().into_iter().find(|l| l.id == "support-r3").unwrap();
        assert_eq!(launch.source, n.root.join("models/support-r3/tune-Q4_0.gguf").display().to_string());
        assert!(!n.root.join("staging").join(&key).exists(), "staging is cleared");

        let list = n.svc.list();
        let rec = list[0].provenance().unwrap();
        assert_eq!((rec.origin.as_str(), rec.uploaded_by.as_deref(), rec.license.as_deref()), (ORIGIN_UPLOAD, Some("laptop"), Some("apache-2.0")));
        assert!(rec.uploaded_ms.is_some());
        assert_eq!(list[0].served_as.as_deref(), Some("support-r3"));
        let disk = client(&n, "laptop").disk("node").await.unwrap();
        assert_eq!((disk.uploaded, disk.staged), (found.size(), 0));
        assert!(disk.free > 0 && disk.total >= disk.free);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn writes_are_refused_with_stable_reasons() {
        let src = tempfile::tempdir().unwrap();
        let file = local(src.path(), "m.gguf", &model_gguf(2000));
        let found = inspect(std::slice::from_ref(&file)).unwrap();
        let up = || Upload::new("m", &found, Provenance::new(ORIGIN_UPLOAD)).serving();
        let files = [file.clone()];

        let refuse_writes: Arc<dyn Authorize> =
            Arc::new(|c: &Caller| if c.peer == "phone" { Err(RemoteError::refused("no write role")) } else { Ok(()) });
        let n = node(EngineKind::Llamacpp, WritePolicy::default().with_max_upload(1000).with_auth(refuse_writes));
        assert_eq!(reason(client(&n, "phone").upload("node", up(), &files, |_| {}).await), refusal::NOT_ALLOWED);
        assert!(matches!(client(&n, "nobody").upload("node", up(), &files, |_| {}).await, Err(RemoteError::Refused { .. })));
        assert_eq!(reason(client(&n, "laptop").upload("node", up(), &files, |_| {}).await), refusal::TOO_BIG);
        assert!(matches!(client(&n, "phone").remove("node", "m").await, Err(e) if refusal_reason(&e) == Some(refusal::NOT_ALLOWED)));

        let n = node(EngineKind::Vllm, WritePolicy::default());
        assert_eq!(reason(client(&n, "laptop").upload("node", up(), &files, |_| {}).await), refusal::WRONG_ENGINE);
        let mut pickle = up();
        pickle.files[0].path = "pytorch_model.bin".into();
        assert_eq!(reason(n.svc.precheck(&pickle).map(|_| unreachable!())), refusal::BAD_NAME);
        let mut bad_id = up();
        bad_id.id = "../x".into();
        assert_eq!(reason(n.svc.precheck(&bad_id).map(|_| unreachable!())), refusal::BAD_NAME);

        let n = node(EngineKind::Llamacpp, WritePolicy::default().with_quota(1000));
        assert_eq!(reason(client(&n, "laptop").upload("node", up(), &files, |_| {}).await), refusal::QUOTA);

        let n = node(EngineKind::Llamacpp, WritePolicy::default());
        let full = ModelsService::new(n.svc.store().clone(), n.sup.clone(), Arc::new(|_: &Caller| Ok(())))
            .with_staging(n.root.join("staging"))
            .with_write_policy(WritePolicy::default().with_keep_free(KeepFree::new(0, u64::MAX / 2)));
        assert_eq!(reason(full.begin(up()).map(|_| unreachable!())), refusal::NO_SPACE);

        // A wrong hash is found on Done; the parts are dropped.
        let mut wrong = up();
        wrong.files[0].sha256 = "ab".repeat(32);
        assert_eq!(reason(client(&n, "laptop").upload("node", wrong.clone(), &files, |_| {}).await), refusal::CHECKSUM);
        assert!(!n.root.join("staging").join(wrong.key()).exists());

        client(&n, "laptop").upload("node", up(), &files, |_| {}).await.unwrap();
        assert_eq!(reason(client(&n, "laptop").upload("node", up(), &files, |_| {}).await), refusal::EXISTS);
        let replaced = client(&n, "laptop").upload("node", up().replacing(), &files, |_| {}).await.unwrap();
        assert!(replaced.served);
        assert_eq!(client(&n, "laptop").remove("node", "m").await.unwrap(), found.size());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn adapters_check_their_base_early_and_serve_on_it() {
        let n = node(EngineKind::Llamacpp, WritePolicy::default());
        let base = local(&n.root, "base.gguf", &base_gguf("llama"));
        n.sup.add_model(ModelLaunch::new("base", EngineKind::Llamacpp, base.display().to_string())).unwrap();
        let src = tempfile::tempdir().unwrap();
        let c = client(&n, "laptop");

        let mut qwen = lora_gguf("qwen3");
        qwen.resize(2 * CHUNK, 0);
        let qwen = local(src.path(), "qwen-LoRA.gguf", &qwen);
        let found = inspect(std::slice::from_ref(&qwen)).unwrap();
        let up = Upload::new("q", &found, Provenance::new(ORIGIN_TRAIN)).serving();
        assert_eq!(reason(c.upload("node", up.clone(), std::slice::from_ref(&qwen), |_| {}).await), refusal::NO_BASE);
        let up = up.with_adapter_for("base");
        assert_eq!(reason(c.upload("node", up.clone(), std::slice::from_ref(&qwen), |_| {}).await), refusal::WRONG_ARCH);
        assert!(!n.root.join("staging").join(up.key()).exists(), "a refused header drops the parts");

        let model = local(src.path(), "not-a-lora.gguf", &model_gguf(2000));
        let found = inspect(std::slice::from_ref(&model)).unwrap();
        let mut as_adapter = Upload::new("x", &found, Provenance::new(ORIGIN_UPLOAD)).with_adapter_for("base").serving();
        as_adapter.provenance.kind = ProvenanceKind::Adapter;
        assert_eq!(reason(c.upload("node", as_adapter, &[model], |_| {}).await), refusal::NOT_ADAPTER);

        let tune = local(src.path(), "tune-LoRA.gguf", &lora_gguf("llama"));
        let found = inspect(std::slice::from_ref(&tune)).unwrap();
        let up = Upload::new("tune", &found, Provenance::new(ORIGIN_TRAIN)).with_adapter_for("base").serving();
        let pulled = c.upload("node", up, &[tune], |_| {}).await.unwrap();
        assert_eq!((pulled.change, pulled.adapter_for.as_deref()), (Some(AdapterChange::Live), Some("base")));
        assert_eq!(pulled.files, ["adapters/tune/tune-LoRA.gguf"]);
        let launch = n.sup.launches().into_iter().find(|l| l.id == "base").unwrap();
        assert_eq!(launch.adapters[0].path, n.root.join("models/adapters/tune/tune-LoRA.gguf").display().to_string());
        let entry = n.svc.list().into_iter().find(|e| e.id() == "tune").unwrap();
        assert_eq!((entry.base.as_deref(), entry.provenance().unwrap().origin.as_str()), (Some("base"), ORIGIN_TRAIN));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_projector_goes_on_its_model_and_comes_off_with_remove() {
        let n = node(EngineKind::Llamacpp, WritePolicy::default());
        let base = local(&n.root, "base.gguf", &base_gguf("llama"));
        n.sup.add_model(ModelLaunch::new("base", EngineKind::Llamacpp, base.display().to_string())).unwrap();
        assert!(!n.sup.launches()[0].capabilities().vision);
        let src = tempfile::tempdir().unwrap();
        let mm = local(src.path(), "mmproj-F16.gguf", &gguf(&[("general.architecture", "clip".into()), ("general.type", "mmproj".into())]));
        let found = inspect(std::slice::from_ref(&mm)).unwrap();
        let up = Upload::new("vision", &found, Provenance::new(ORIGIN_UPLOAD)).with_projector_for("base").serving();
        let pulled = client(&n, "laptop").upload("node", up, &[mm], |_| {}).await.unwrap();
        assert_eq!((pulled.change, pulled.projector_for.as_deref()), (Some(AdapterChange::Live), Some("base")));
        let launch = &n.sup.launches()[0];
        assert_eq!(launch.mmproj.as_deref(), Some(n.root.join("models/projectors/vision/mmproj-F16.gguf").display().to_string().as_str()));
        assert!(launch.capabilities().vision, "the projector makes it take images");
        let entry = n.svc.list().into_iter().find(|e| e.id() == "vision").unwrap();
        assert_eq!(entry.base.as_deref(), Some("base"));
        assert_eq!(client(&n, "laptop").remove("node", "vision").await.unwrap(), found.size());
        assert_eq!(n.sup.launches()[0].mmproj, None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_safetensors_model_serves_on_vllm_from_its_directory() {
        let n = node(EngineKind::Vllm, WritePolicy::default());
        let src = tempfile::tempdir().unwrap();
        let dir = src.path().join("hf");
        local(&dir, "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#);
        let header = serde_json::to_vec(&json!({"w": {"dtype": "U8", "shape": [100], "data_offsets": [0, 100]}})).unwrap();
        let mut st = (header.len() as u64).to_le_bytes().to_vec();
        st.extend(header);
        st.extend([3u8; 100]);
        local(&dir, "model.safetensors", &st);
        let found = inspect(std::slice::from_ref(&dir)).unwrap();
        let up = Upload::new("qwen-tune", &found, Provenance::new(ORIGIN_UPLOAD)).serving();
        let pulled = client(&n, "laptop").upload("node", up, &[dir], |_| {}).await.unwrap();
        assert!(pulled.served);
        let launch = n.sup.launches().into_iter().find(|l| l.id == "qwen-tune").unwrap();
        assert_eq!((launch.engine, launch.source.clone()), (EngineKind::Vllm, n.root.join("models/safetensors/qwen-tune").display().to_string()));
        let entry = &n.svc.list()[0];
        assert_eq!((entry.model.as_ref().unwrap().layout, entry.served_as.as_deref()), (ModelLayout::Safetensors, Some("qwen-tune")));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn nodes_copy_an_upload_from_each_other() {
        let a = node(EngineKind::Llamacpp, WritePolicy::default());
        let to_a = MemoryDialer::new();
        route(&to_a, "a", a.svc.clone(), "node-b");
        let b0 = node(EngineKind::Llamacpp, WritePolicy::default());
        let b = Node {
            svc: Arc::new(
                ModelsService::new(b0.svc.store().clone(), b0.sup.clone(), Arc::new(|_: &Caller| Ok(())))
                    .with_staging(b0.root.join("staging"))
                    .with_write_policy(WritePolicy::default().with_keep_free(KeepFree::none()))
                    .with_peer_dialer(Arc::new(to_a), None),
            ),
            ..b0
        };
        let owner = MemoryDialer::new();
        route(&owner, "a", a.svc.clone(), "laptop");
        route(&owner, "b", b.svc.clone(), "laptop");
        let owner = ModelsClient::new(Arc::new(owner));

        let src = tempfile::tempdir().unwrap();
        let file = local(src.path(), "m.gguf", &model_gguf(2 * CHUNK + 7));
        let found = inspect(std::slice::from_ref(&file)).unwrap();
        let up = Upload::new("m-r1", &found, Provenance::new(ORIGIN_UPLOAD).with_license("mit")).serving();
        owner.upload("a", up.clone(), &[file], |_| {}).await.unwrap();

        let progress = Mutex::new(Vec::new());
        let results = owner.replicate("a", &["b".to_string()], &up, |t, p| progress.lock().unwrap().push((t.to_string(), p.done))).await;
        let (target, r) = &results[0];
        let pulled = r.as_ref().unwrap();
        assert_eq!((target.as_str(), pulled.repo.as_str(), pulled.served), ("b", "a", true));
        assert_eq!(pulled.revision.as_deref(), Some(found.revision.as_str()));
        assert!(progress.lock().unwrap().iter().any(|(t, d)| t == "b" && *d == found.size()));
        let rec = b.svc.list()[0].provenance().cloned().unwrap();
        assert_eq!((rec.license.as_deref(), rec.uploaded_by.as_deref()), (Some("mit"), Some("laptop")), "the record travels with the files");
        assert!(b.sup.launches().iter().any(|l| l.id == "m-r1"));

        let err = owner.pull("b", Pull::from_peer("a", "nope"), |_| {}).await.unwrap_err();
        assert_eq!(refusal_reason(&err), Some(refusal::NOT_FOUND));
        let err = owner.pull("a", Pull::from_peer("b", "m-r1"), |_| {}).await.unwrap_err();
        assert!(matches!(err, RemoteError::Unavailable { .. }), "a has no peer dialer: {err:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn abandoned_and_stale_parts_are_removed() {
        let n = node(EngineKind::Llamacpp, WritePolicy::default().with_part_ttl(Duration::from_millis(1)));
        let staged = n.root.join("staging/0123456789abcdef/files");
        std::fs::create_dir_all(&staged).unwrap();
        std::fs::write(staged.join("x.gguf.part"), b"abc").unwrap();
        assert_eq!(client(&n, "laptop").disk("node").await.unwrap().staged, 3);
        assert!(client(&n, "laptop").abandon_upload("node", "0123456789abcdef").await.unwrap());
        assert!(!client(&n, "laptop").abandon_upload("node", "0123456789abcdef").await.unwrap());
        assert!(client(&n, "laptop").abandon_upload("node", "../x").await.is_err());
        std::fs::create_dir_all(&staged).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(n.svc.sweep_uploads(), 1);
    }

    #[test]
    fn keep_free_takes_the_larger_and_reasons_parse() {
        assert_eq!(KeepFree::default().of(1000 << 30), 100 << 30);
        assert_eq!(KeepFree::default().of(100 << 30), 20 << 30);
        assert_eq!(KeepFree::none().of(u64::MAX), 0);
        assert_eq!(refusal_reason(&refuse(refusal::QUOTA, "x")), Some("quota"));
        assert_eq!(refusal_reason(&RemoteError::refused("owner only")), None);
        assert_eq!(refusal_reason(&RemoteError::Busy), None);
    }

    fn recorder() -> (Arc<Mutex<Vec<Placed>>>, impl Fn(&Placed) + Send + Sync + 'static) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        (seen, move |p: &Placed| s.lock().unwrap().push(p.clone()))
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn placement_hooks_hear_uploads_and_peer_copies() {
        let (on_a, hook_a) = recorder();
        let a = node_with(EngineKind::Llamacpp, WritePolicy::default(), |s| s.on_placed(|_| panic!("a broken hook")).on_placed(hook_a));
        let to_a = MemoryDialer::new();
        route(&to_a, "a", a.svc.clone(), "node-b");
        let (on_b, hook_b) = recorder();
        let b = node_with(EngineKind::Llamacpp, WritePolicy::default(), |s| s.with_peer_dialer(Arc::new(to_a), None).on_placed(hook_b));
        let owner = MemoryDialer::new();
        route(&owner, "a", a.svc.clone(), "laptop");
        route(&owner, "b", b.svc.clone(), "laptop");
        let owner = ModelsClient::new(Arc::new(owner));

        let src = tempfile::tempdir().unwrap();
        let file = local(src.path(), "tune-Q4_0.gguf", &model_gguf(CHUNK + 5));
        let found = inspect(std::slice::from_ref(&file)).unwrap();
        let up = Upload::new("tune", &found, Provenance::new(ORIGIN_TRAIN).with_license("mit")).serving();
        let pulled = owner.upload("a", up.clone(), &[file], |_| {}).await.unwrap();

        let seen = on_a.lock().unwrap().clone();
        assert_eq!(seen.len(), 1, "one placement, despite the panicking hook");
        let p = &seen[0];
        assert_eq!((p.id(), p.via.clone(), p.revision()), ("tune", PlacedVia::Upload, Some(found.revision.as_str())));
        assert_eq!(p.pulled, pulled);
        assert_eq!((p.served_as(), p.quant().as_deref(), p.serve_error.clone()), (Some("tune"), Some("Q4_0"), None));
        let rec = p.provenance().unwrap();
        assert_eq!((rec.origin.as_str(), rec.license.as_deref(), rec.uploaded_by.as_deref()), (ORIGIN_TRAIN, Some("mit"), Some("laptop")));
        assert_eq!(p.entry.model.as_ref().unwrap().files, ["tune/tune-Q4_0.gguf"]);

        owner.replicate("a", &["b".to_string()], &up, |_, _| {}).await[0].1.as_ref().unwrap();
        let seen = on_b.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!((seen[0].via.clone(), seen[0].revision()), (PlacedVia::Peer { node: "a".into() }, Some(found.revision.as_str())));
        assert_eq!(seen[0].provenance().unwrap().license.as_deref(), Some("mit"));
        assert_eq!(on_a.lock().unwrap().len(), 1, "serving a fetch places nothing");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hooks_hear_adapters_projectors_and_placements_that_fail_to_serve() {
        let (seen, hook) = recorder();
        let n = node_with(EngineKind::Llamacpp, WritePolicy::default(), |s| s.on_placed(hook));
        let base = local(&n.root, "base.gguf", &base_gguf("llama"));
        n.sup.add_model(ModelLaunch::new("base", EngineKind::Llamacpp, base.display().to_string())).unwrap();
        let src = tempfile::tempdir().unwrap();
        let c = client(&n, "laptop");

        let lora = local(src.path(), "t-LoRA.gguf", &lora_gguf("llama"));
        let found = inspect(std::slice::from_ref(&lora)).unwrap();
        c.upload("node", Upload::new("t", &found, Provenance::new(ORIGIN_UPLOAD)).with_adapter_for("base").serving(), &[lora], |_| {}).await.unwrap();
        let mm = local(src.path(), "mmproj.gguf", &gguf(&[("general.architecture", "clip".into())]));
        let found = inspect(std::slice::from_ref(&mm)).unwrap();
        c.upload("node", Upload::new("eyes", &found, Provenance::new(ORIGIN_UPLOAD)).with_projector_for("base").serving(), &[mm], |_| {})
            .await
            .unwrap();
        // An embedding model takes no projector: the import works, serving fails.
        let mut emb = ModelLaunch::new("emb", EngineKind::Llamacpp, base.display().to_string());
        emb.embedding = true;
        n.sup.add_model(emb).unwrap();
        let mm = local(src.path(), "mmproj2.gguf", &gguf(&[("general.architecture", "clip".into())]));
        let found = inspect(std::slice::from_ref(&mm)).unwrap();
        let up = Upload::new("eyes2", &found, Provenance::new(ORIGIN_UPLOAD)).with_projector_for("emb").serving();
        let err = c.upload("node", up, &[mm], |_| {}).await.unwrap_err();

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 3);
        let adapter = seen[0].entry.adapter.as_ref().unwrap();
        assert_eq!((adapter.id.as_str(), seen[0].entry.base.as_deref(), seen[0].quant()), ("t", Some("base"), None));
        assert!(seen[0].entry.model.is_none());
        assert_eq!((seen[1].entry.projector.as_ref().unwrap().id.as_str(), seen[1].entry.base.as_deref()), ("eyes", Some("base")));
        assert_eq!(seen[1].pulled.projector_for.as_deref(), Some("base"));
        assert_eq!((seen[2].id(), seen[2].pulled.served, seen[2].serve_error.as_ref()), ("eyes2", false, Some(&err)));
        assert!(seen[2].entry.projector.is_some(), "placed, though not served");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn node_info_follows_placements() {
        use weirdnet_net::Keypair;

        use crate::host::ModelHost;
        use crate::node::NodeService;

        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let info = Arc::new(NodeService::new(Arc::new(ModelHost::new(sup.clone(), auth.clone())), Keypair::generate_ed25519(), None, auth));
        let tmp = tempfile::tempdir().unwrap();
        let svc = ModelsService::new(ModelStore::new(tmp.path().join("models")), sup, Arc::new(|_: &Caller| Ok(())))
            .with_staging(tmp.path().join("staging"))
            .with_write_policy(WritePolicy::default().with_keep_free(KeepFree::none()))
            .with_node_info(info.clone());
        let d = MemoryDialer::new();
        route(&d, "node", Arc::new(svc), "laptop");
        let c = ModelsClient::new(Arc::new(d));

        let file = local(tmp.path(), "src/m-Q8_0.gguf", &model_gguf(100));
        let found = inspect(std::slice::from_ref(&file)).unwrap();
        c.upload("node", Upload::new("m", &found, Provenance::new(ORIGIN_UPLOAD).with_license("Apache-2.0")).serving(), &[file], |_| {})
            .await
            .unwrap();
        let cap = info.capability();
        let entry = cap.models.iter().find(|m| m.status.id == "m").unwrap();
        assert_eq!((entry.revision.as_deref(), entry.quant.as_deref()), (Some(found.revision.as_str()), Some("Q8_0")));
        assert_eq!((entry.origin.as_deref(), entry.license.as_deref()), (Some("upload"), Some("Apache-2.0")));
    }
}
