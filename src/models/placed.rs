//! Placement hooks: the app hears about every entry a [`ModelsService`] puts in its store.
//!
//! [`ModelsService::on_placed`] registers a callback that runs after an upload, a copy from
//! another node ([`PullSource::Peer`](super::PullSource::Peer), which is also what [`ModelsClient::replicate`](super::ModelsClient::replicate) makes each
//! target do) or a Hub pull has placed its files (and served them, when asked). It gets a
//! [`Placed`]: the stored entry as `List` would show it (with its provenance), the
//! [`Pulled`] answer (with the content revision) and how it arrived. Apps use it to update their
//! own records and the node's capability record without a protocol of their own;
//! [`ModelsService::with_node_info`] does the capability part.
//!
//! Uploads and peer copies fire once their files are placed, even when serving them then fails
//! ([`Placed::serve_error`]); Hub pulls fire when the pull succeeds.
//!
//! Callbacks run on the task that placed the entry, before the client gets its answer, so they
//! should be quick: hand slow work to a task or a channel. A panicking callback is caught and
//! does not fail the placement.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use weirdnet_agent::remote::RemoteError;
use weirdnet_model::artifact::Provenance;
use weirdnet_model::{Imported, ModelLayout};

use super::{ModelsService, Pulled, StoredEntry};
use crate::node::NodeService;

/// How a placed entry arrived.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlacedVia {
    /// Uploaded by a client ([`super::ModelsCommand::Upload`]).
    Upload,
    /// Copied from node `node` ([`PullSource::Peer`](super::PullSource::Peer)).
    Peer { node: String },
    /// Downloaded from Hub repo `repo` ([`super::ModelsCommand::Pull`]).
    Hub { repo: String },
}

/// One entry placed in the store; see [`ModelsService::on_placed`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Placed {
    /// The stored model, adapter or projector, with its provenance and serving state.
    pub entry: StoredEntry,
    /// The answer the client gets (files, size, whether it is served, the revision).
    pub pulled: Pulled,
    pub via: PlacedVia,
    /// Set when the files were placed but serving them failed (the client gets this error);
    /// the entry stays in the store.
    pub serve_error: Option<RemoteError>,
}

impl Placed {
    /// The entry's id in the store.
    pub fn id(&self) -> &str {
        &self.pulled.id
    }

    /// The content revision (uploads and peer copies; `None` for Hub pulls).
    pub fn revision(&self) -> Option<&str> {
        self.pulled.revision.as_deref()
    }

    /// Where it came from, for uploaded (and trained) entries.
    pub fn provenance(&self) -> Option<&Provenance> {
        self.entry.provenance()
    }

    /// The supervisor id it is served under, if it is served.
    pub fn served_as(&self) -> Option<&str> {
        self.entry.served_as.as_deref().or(self.pulled.served.then_some(self.pulled.id.as_str()))
    }

    /// The quantization in a GGUF model's file name (`Q4_K_M`), if any.
    pub fn quant(&self) -> Option<String> {
        let model = self.entry.model.as_ref().filter(|m| m.layout == ModelLayout::Gguf)?;
        let name = model.files.first()?.rsplit('/').next()?;
        weirdnet_model::hf::quant_of(name)
    }
}

/// A placement callback ([`ModelsService::on_placed`]).
pub type OnPlaced = Arc<dyn Fn(&Placed) + Send + Sync>;

impl ModelsService {
    /// Calls `hook` after each upload, peer copy or Hub pull is placed (see [`Placed`]).
    /// Several hooks run in the order they were added.
    pub fn on_placed(mut self, hook: impl Fn(&Placed) + Send + Sync + 'static) -> Self {
        self.placed.push(Arc::new(hook));
        self
    }

    /// Keeps `node`'s capability record current: each placed entry with a revision (uploads and
    /// peer copies) goes to [`NodeService::set_model_info`] under the id it is served as (its
    /// store id when it is not served), with the quantization from a GGUF model's file name.
    /// Hub pulls carry no revision and leave the record alone.
    pub fn with_node_info(self, node: Arc<NodeService>) -> Self {
        self.on_placed(move |p| {
            if let Some(revision) = p.revision() {
                let id = p.served_as().unwrap_or(p.id()).to_string();
                node.set_model_info(&id, Some(revision.to_string()), p.quant());
            }
        })
    }

    /// Runs the placement hooks for `pulled`, finding its entry in the store.
    pub(super) fn fire_placed(&self, pulled: &Pulled, via: PlacedVia, imported: Option<&Imported>, serve_error: Option<&RemoteError>) {
        if self.placed.is_empty() {
            return;
        }
        let Some(entry) = self.placed_entry(pulled, imported) else { return };
        let placed = Placed { entry, pulled: pulled.clone(), via, serve_error: serve_error.cloned() };
        for hook in &self.placed {
            let _ = catch_unwind(AssertUnwindSafe(|| hook(&placed)));
        }
    }

    fn placed_entry(&self, pulled: &Pulled, imported: Option<&Imported>) -> Option<StoredEntry> {
        let projector = pulled.projector_for.is_some() || matches!(imported, Some(Imported::Projector(_)));
        let adapter = pulled.adapter_for.is_some() || matches!(imported, Some(Imported::Adapter(_)));
        let first = pulled.files.first();
        let found = self.list().into_iter().find(|e| match (&e.model, &e.adapter, &e.projector) {
            (Some(m), _, _) if !adapter && !projector => m.id == pulled.id || first.is_some_and(|f| m.files.first() == Some(f)),
            (_, Some(a), _) if adapter => a.id == pulled.id,
            (_, _, Some(p)) if projector => p.id == pulled.id,
            _ => false,
        });
        found.or_else(|| match imported? {
            Imported::Model(m) => Some(StoredEntry::new(Some(m.clone()), None, None)),
            Imported::Adapter(a) => Some(StoredEntry::new(None, Some(a.clone()), None)),
            Imported::Projector(p) => Some(StoredEntry::new(None, None, Some(p.clone()))),
            _ => None,
        })
    }
}
