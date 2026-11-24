//! `/weirdnet-backends/node/1`: how inference nodes drive model nodes over `weirdnet-net`.
//!
//! A model node pushes a [`Capability`] record to every subscriber: signed by the node's key,
//! chained to the user by its [`NodeCert`], with a sequence number and a 90 second expiry,
//! refreshed every 15 seconds and on every change. It takes [`Command`]s: `Load`, `Unload`,
//! `Drain`, `Reserve`/`Release` (hold the node's GPU for one job: nothing is placed there until
//! released; nodes before 0.4.3 fail these calls), `GroupUp`/`GroupDown` and `Health`. Inference itself is raw infer/2 framing through
//! [`ModelHost`], served next to node/1 by [`NodeServer`] on [`RAW_INFER_PROTOCOL`]
//! (`/weirdnet-backends/infer/1`) and, unless [`NodeServerOptions::legacy_infer2`] is off, on
//! `/weirdnet-agent/infer/2` as in 0.1.0. Inference nodes dial model nodes through
//! [`RawInferDialer`](crate::RawInferDialer), which prefers the new id.
//!
//! Each stream carries one [`ClientMsg`] (CBOR, with the caller's certificate) and its answers
//! ([`ServerMsg`]); a `Subscribe` stream stays open for records until either side closes it or
//! the node withdraws. Which nodes may drive a model node is the [`Authorize`] hook's decision
//! (`weirdnet_agent::serve::RosterAuth` admits the user's linked devices).
//!
//! Swap groups that span machines (a Ray worker or an `rpc-server` bound to a private link)
//! are started by the application's [`GroupWorkers`]; without one, a node answers `GroupUp`
//! with `Unsupported` and says `group_jobs: false` in its record.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use weirdnet_agent::remote::infer::{Admission, InferHost};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError, from_cbor, to_cbor};
use weirdnet_agent::serve::{Server, Serving, channel, stream_limits};
use weirdnet_model::hw::{Budget, Hardware};
use weirdnet_net::node::{Message, Node};
use weirdnet_net::profile::Profile;
use weirdnet_net::user::NodeCert;
use weirdnet_net::{AppBody, AppResponse, Keypair, PeerId, PublicKey};

use crate::engine::EngineKind;
use crate::host::{ModelHost, RAW_INFER_PROTOCOL, Speed};
pub use crate::link::{AddrClass, LinkInfo, LinkMeasure};
use crate::supervisor::{ModelState, ModelStatus, Reservation, SupervisorEvent};
use weirdnet_agent::serve::RosterView;

/// The protocol id.
pub const PROTOCOL: &str = "/weirdnet-backends/node/1";
/// The app name model nodes announce (`Node::serve_app`), so inference nodes find them among
/// the user's linked devices.
pub const MODEL_NODE_APP: &str = "weirdnet-backends-node";
/// How long a record is good for.
pub const RECORD_TTL: Duration = Duration::from_secs(90);
/// How often a subscriber gets a fresh record when nothing changed.
pub const REFRESH: Duration = Duration::from_secs(15);
/// Accepted clock skew on a record's issue time.
const SKEW_MS: i64 = 60_000;
/// Signatures cover this, then the record bytes.
const DOMAIN: &[u8] = b"weirdnet-backends/node/1 capability record\n";
/// Recent supervisor events kept for health reports.
const RECENT: usize = 32;

fn now_ms() -> i64 {
    weirdnet_net::now_ms()
}

/// What a model node can do right now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Capability {
    /// The node's peer id; the record is signed by its key.
    pub peer: String,
    /// Increases with every record the node issues.
    pub seq: u64,
    pub issued_ms: i64,
    pub expires_ms: i64,
    /// Accelerators, memory and interconnects.
    pub hardware: Option<Hardware>,
    /// Memory models may use, and how much is free.
    pub budget: Option<Budget>,
    /// Engines the node can run.
    pub engines: Vec<EngineKind>,
    /// Every model the node is configured for: loaded or loadable from disk.
    pub models: Vec<ModelEntry>,
    pub load: Load,
    /// Hashes of prompt prefixes the engines hold warm (supplied by the application).
    pub warm_prefixes: Vec<String>,
    /// Swap groups this node is in.
    pub groups: Vec<String>,
    /// Draining: no new requests.
    pub draining: bool,
    /// Takes group jobs (`GroupUp`).
    pub group_jobs: bool,
    /// Reserved for a job ([`Command::Reserve`]): it runs no model, and nothing should be
    /// placed on it until it is released. Absent when not reserved, and in records from nodes
    /// before 0.4.3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved: Option<Reservation>,
    /// Private interfaces (never public or CGNAT addresses), for placing swap groups. Absent
    /// in records from 0.1.1 nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<LinkInfo>,
}

/// One model in a [`Capability`] record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelEntry {
    pub status: ModelStatus,
    pub revision: Option<String>,
    pub quant: Option<String>,
    /// The last measured load time, else a size-based guess.
    pub load_secs_estimate: Option<f32>,
    pub speed: Speed,
}

/// The node's load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Load {
    /// Requests waiting for a slot (all priorities).
    pub queued: u32,
    /// Requests holding a slot.
    pub active: u32,
    /// Concurrent request slots.
    pub slots: u32,
    /// Requests under way in the engines.
    pub in_flight: u32,
}

/// A capability record with its signature and certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SignedRecord {
    /// The CBOR [`Capability`].
    #[serde(with = "serde_bytes")]
    pub record: Vec<u8>,
    /// The node's public key (protobuf).
    #[serde(with = "serde_bytes")]
    pub public_key: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
    /// The user's certificate for the node.
    pub cert: Option<NodeCert>,
}

fn signed_bytes(record: &[u8]) -> Vec<u8> {
    [DOMAIN, record].concat()
}

impl SignedRecord {
    /// Signs `cap` with the node's key.
    pub fn sign(keypair: &Keypair, cap: &Capability, cert: Option<NodeCert>) -> Result<Self, RemoteError> {
        let record = to_cbor(cap)?;
        let signature = keypair.sign(&signed_bytes(&record)).map_err(|e| RemoteError::Failed { message: format!("signing: {e}") })?;
        Ok(Self { record, public_key: keypair.public().encode_protobuf(), signature, cert })
    }

    /// Checks the signature, that the record is `from_peer`'s own and that it has not expired.
    /// Says nothing about the user: see [`Verifier`].
    pub fn verify(&self, from_peer: &str, now_ms: i64) -> Result<Capability, RemoteError> {
        let bad = |why: &str| RemoteError::protocol(format!("capability record: {why}"));
        let key = PublicKey::try_decode_protobuf(&self.public_key).map_err(|_| bad("bad public key"))?;
        if PeerId::from_public_key(&key).to_string() != from_peer {
            return Err(bad("signed by another node"));
        }
        if !key.verify(&signed_bytes(&self.record), &self.signature) {
            return Err(bad("bad signature"));
        }
        let cap: Capability = from_cbor(&self.record)?;
        if cap.peer != from_peer {
            return Err(bad("names another node"));
        }
        if cap.expires_ms <= now_ms {
            return Err(bad("expired"));
        }
        if cap.issued_ms > now_ms + SKEW_MS {
            return Err(bad("issued in the future"));
        }
        Ok(cap)
    }
}

/// Checks records against the user: valid signature, and a valid, unexpired certificate from
/// the user for the very node that signed. With [`with_roster`](Self::with_roster) the node
/// must also be linked in the roster now, so a revoked node presenting its old certificate is
/// refused.
#[derive(Clone)]
pub struct Verifier {
    profile: &'static Profile,
    user: Option<PublicKey>,
    roster: Option<RosterView>,
}

impl Verifier {
    pub fn new(profile: &'static Profile, user: PublicKey) -> Self {
        Self { profile, user: Some(user), roster: None }
    }

    /// Signature and expiry only (tests, or when the dialer already checked the peer).
    pub fn signature_only(profile: &'static Profile) -> Self {
        Self { profile, user: None, roster: None }
    }

    /// Also checks the record's node against the roster (`Roster::check_caller`): unknown,
    /// revoked and wrong-user nodes are refused. Data-key admission doesn't matter here.
    pub fn with_roster(mut self, roster: RosterView) -> Self {
        self.roster = Some(roster);
        self
    }

    pub fn verify(&self, record: &SignedRecord, from_peer: &str, now_ms: i64) -> Result<Capability, RemoteError> {
        let cap = record.verify(from_peer, now_ms)?;
        if let Some(user) = &self.user {
            let ok = match &record.cert {
                Some(c) => c.peer_id == from_peer && c.active() && self.profile.verify_cert(user, c),
                None => false,
            };
            if !ok {
                return Err(RemoteError::refused("capability record: no valid certificate from the user"));
            }
            if let Some(roster) = &self.roster {
                use weirdnet_sync::CallerCheck;
                match roster().check_caller(self.profile, user, from_peer, record.cert.as_ref()) {
                    CallerCheck::Linked | CallerCheck::NotAdmitted => {}
                    other => {
                        let reason = other.reason().unwrap_or("unknown-device");
                        return Err(RemoteError::refused(format!("capability record: {reason}")));
                    }
                }
            }
        }
        Ok(cap)
    }
}

/// A swap group spanning machines, as the coordinator planned it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GroupPlan {
    pub group: String,
    /// The peer that serves the group's model.
    pub leader: String,
    pub members: Vec<String>,
    /// Models to unload here first.
    pub drop: Vec<String>,
    /// The model the leader serves.
    pub model: Option<String>,
    /// The private link address workers bind to.
    pub link: Option<String>,
    /// Worker kind, for example `ray` or `rpc-server`.
    pub worker: Option<String>,
    /// Anything else the application's workers need.
    pub extra: HashMap<String, String>,
}

impl GroupPlan {
    pub fn new(group: impl Into<String>, leader: impl Into<String>) -> Self {
        Self { group: group.into(), leader: leader.into(), ..Default::default() }
    }

    pub fn with_members(mut self, members: Vec<String>) -> Self {
        self.members = members;
        self
    }

    pub fn with_drop(mut self, drop: Vec<String>) -> Self {
        self.drop = drop;
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_link(mut self, link: impl Into<String>) -> Self {
        self.link = Some(link.into());
        self
    }

    pub fn with_worker(mut self, worker: impl Into<String>) -> Self {
        self.worker = Some(worker.into());
        self
    }
}

/// Starts and stops the processes of a cross-machine swap group (a Ray worker, an
/// `rpc-server`). The application provides it; this crate has no such adapters.
#[async_trait]
pub trait GroupWorkers: Send + Sync {
    /// Join `plan` as `me` (the leader or a member), after the plan's `drop` models unloaded.
    async fn up(&self, me: &str, plan: &GroupPlan) -> Result<(), RemoteError>;
    async fn down(&self, group: &str) -> Result<(), RemoteError>;
}

/// A node/1 request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Command {
    /// Records now and on every change, until either side closes.
    Subscribe,
    /// One record.
    Record,
    Load {
        model: String,
        force: bool,
    },
    Unload {
        model: String,
    },
    /// Stop taking requests (`on`) and wait for the ones under way; or take them again.
    Drain {
        on: bool,
    },
    /// Hold the node's GPU for `holder`: drain and stop every engine and start none (loads,
    /// requests and group jobs are refused) until `Release` by the same holder
    /// ([`Supervisor::reserve`](crate::Supervisor::reserve)). Since 0.4.3.
    Reserve {
        holder: String,
    },
    /// End `holder`'s reservation. Since 0.4.3.
    Release {
        holder: String,
    },
    GroupUp {
        plan: GroupPlan,
    },
    GroupDown {
        group: String,
    },
    Health,
}

/// What a client sends: its certificate and one command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientMsg {
    /// The caller's CBOR `NodeCert`, for the authorize hook.
    #[serde(default, with = "serde_bytes")]
    pub cert: Option<Vec<u8>>,
    pub command: Command,
}

/// What a model node answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ServerMsg {
    Record(SignedRecord),
    /// The command succeeded.
    Done,
    Health(Health),
    Failed(RemoteError),
    /// The node is leaving; its records are void.
    Withdrawn {
        peer: String,
        seq: u64,
    },
}

/// A health report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Health {
    pub peer: String,
    pub at_ms: i64,
    /// No model failed or in a crash loop.
    pub ok: bool,
    pub draining: bool,
    /// The node's reservation (since 0.4.3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved: Option<Reservation>,
    pub in_flight: u32,
    pub uptime_secs: u64,
    pub models: Vec<ModelHealth>,
    /// Recent supervisor events, oldest first.
    pub events: Vec<SupervisorEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelHealth {
    pub id: String,
    pub state: ModelState,
    pub crashes: usize,
}

/// A model's revision and quantization.
type ModelInfo = (Option<String>, Option<String>);

/// A model node's node/1 side: issues records and runs commands against its [`ModelHost`].
/// Transport-neutral ([`NodeService::serve`] takes a channel); [`NodeServer`] runs it on a
/// `weirdnet-net` node.
pub struct NodeService {
    host: Arc<ModelHost>,
    admission: Admission,
    slots: usize,
    keypair: Keypair,
    peer: String,
    cert: Option<NodeCert>,
    auth: Arc<dyn Authorize>,
    hardware: Option<Hardware>,
    budget: Option<Budget>,
    engines: Vec<EngineKind>,
    workers: Option<Arc<dyn GroupWorkers>>,
    drain_timeout: Duration,
    info: Mutex<HashMap<String, ModelInfo>>,
    warm: Mutex<Vec<String>>,
    groups: Mutex<Vec<String>>,
    links: Mutex<Vec<LinkInfo>>,
    recent: Mutex<VecDeque<SupervisorEvent>>,
    seq: AtomicU64,
    withdrawn: AtomicBool,
    changed: watch::Sender<u64>,
    started: Instant,
}

impl NodeService {
    /// `keypair` is the node's own (records are signed with it); `cert` the user's certificate
    /// for it; `auth` decides who may drive the node.
    pub fn new(host: Arc<ModelHost>, keypair: Keypair, cert: Option<NodeCert>, auth: Arc<dyn Authorize>) -> Self {
        let peer = keypair.public().to_peer_id().to_string();
        let engines = host.supervisor().status().iter().map(|m| m.engine).fold(Vec::new(), |mut v, e| {
            if !v.contains(&e) {
                v.push(e);
            }
            v
        });
        Self {
            host,
            admission: Admission::new(4),
            slots: 4,
            keypair,
            peer,
            cert,
            auth,
            hardware: None,
            budget: None,
            engines,
            workers: None,
            drain_timeout: Duration::from_secs(60),
            info: Mutex::default(),
            warm: Mutex::default(),
            groups: Mutex::default(),
            links: Mutex::default(),
            recent: Mutex::default(),
            seq: AtomicU64::new(now_ms().max(0) as u64),
            withdrawn: AtomicBool::new(false),
            changed: watch::Sender::new(0),
            started: Instant::now(),
        }
    }

    /// Infer/2 admission: `slots` concurrent requests, `queue` waiting (default 4 and 0).
    pub fn with_admission(mut self, slots: usize, queue: usize) -> Self {
        self.admission = Admission::with_queue(slots, queue);
        self.slots = slots;
        self
    }

    /// The hardware and memory budget for the record (`weirdnet_model::hw::detect` and `budget`).
    pub fn with_hardware(mut self, hardware: Hardware, budget: Budget) -> Self {
        self.hardware = Some(hardware);
        self.budget = Some(budget);
        self
    }

    pub fn with_group_workers(mut self, workers: Arc<dyn GroupWorkers>) -> Self {
        self.workers = Some(workers);
        self
    }

    /// How long `Drain` waits for requests under way (default 60 s).
    pub fn with_drain_timeout(mut self, d: Duration) -> Self {
        self.drain_timeout = d;
        self
    }

    pub fn peer(&self) -> &str {
        &self.peer
    }

    pub fn host(&self) -> &Arc<ModelHost> {
        &self.host
    }

    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    /// A model's revision and quantization, for the record.
    pub fn set_model_info(&self, model: &str, revision: Option<String>, quant: Option<String>) {
        self.info.lock().unwrap_or_else(|e| e.into_inner()).insert(model.into(), (revision, quant));
        self.touch();
    }

    /// The private links for the record (for example `link::local_links()` with the operator's
    /// declared speeds). Public and CGNAT addresses are dropped.
    pub fn set_links(&self, links: Vec<LinkInfo>) {
        *self.links.lock().unwrap_or_else(|e| e.into_inner()) = crate::link::advertised(links);
        self.touch();
    }

    /// Prefix hashes the engines hold warm.
    pub fn set_warm_prefixes(&self, hashes: Vec<String>) {
        *self.warm.lock().unwrap_or_else(|e| e.into_inner()) = hashes;
        self.touch();
    }

    /// Something in the record changed: subscribers get a fresh one.
    pub fn touch(&self) {
        self.changed.send_modify(|v| *v += 1);
    }

    /// The node is leaving: subscribers get `Withdrawn` and their streams end.
    pub fn withdraw(&self) {
        self.withdrawn.store(true, Ordering::SeqCst);
        self.touch();
    }

    /// The current capability.
    pub fn capability(&self) -> Capability {
        let issued_ms = now_ms();
        let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let info = self.info.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let models = self
            .host
            .supervisor()
            .status()
            .into_iter()
            .map(|status| {
                let (revision, quant) = info.get(&status.id).cloned().unwrap_or_default();
                // Without a measured load, assume about 1 GB/s from disk plus a few seconds.
                let load_secs_estimate = status.last_load_secs.or(status.size_bytes.map(|b| b as f32 / 1e9 + 3.0));
                let speed = self.host.speed(&status.id);
                ModelEntry { status, revision, quant, load_secs_estimate, speed }
            })
            .collect::<Vec<_>>();
        let in_flight = self.host.supervisor().in_flight() as u32;
        let free = self.admission.free();
        let load =
            Load { queued: self.admission.waiting() as u32, active: self.slots.saturating_sub(free) as u32, slots: self.slots as u32, in_flight };
        Capability {
            peer: self.peer.clone(),
            seq,
            issued_ms,
            expires_ms: issued_ms + RECORD_TTL.as_millis() as i64,
            hardware: self.hardware.clone(),
            budget: self.budget,
            engines: self.engines.clone(),
            models,
            load,
            warm_prefixes: self.warm.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            groups: self.groups.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            draining: self.host.draining(),
            group_jobs: self.workers.is_some(),
            reserved: self.host.supervisor().reservation(),
            links: self.links.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    }

    /// The current capability, signed.
    pub fn record(&self) -> Result<SignedRecord, RemoteError> {
        SignedRecord::sign(&self.keypair, &self.capability(), self.cert.clone())
    }

    pub fn health(&self) -> Health {
        let status = self.host.supervisor().status();
        let ok = status.iter().all(|m| !matches!(m.state, ModelState::Failed(_)));
        Health {
            peer: self.peer.clone(),
            at_ms: now_ms(),
            ok,
            draining: self.host.draining(),
            reserved: self.host.supervisor().reservation(),
            in_flight: self.host.supervisor().in_flight() as u32,
            uptime_secs: self.started.elapsed().as_secs(),
            models: status.into_iter().map(|m| ModelHealth { id: m.id, state: m.state, crashes: m.crashes }).collect(),
            events: self.recent.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect(),
        }
    }

    /// Keeps the record current with the supervisor's events. Runs until the supervisor goes.
    pub fn watch(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        let mut events = self.host.supervisor().subscribe();
        let mut reserved = self.host.supervisor().reservation_changes();
        tokio::spawn(async move {
            loop {
                let ev = tokio::select! {
                    ev = events.recv() => ev,
                    r = reserved.changed() => {
                        if r.is_err() {
                            return;
                        }
                        me.touch();
                        continue;
                    }
                };
                match ev {
                    Ok(ev) => {
                        let mut recent = me.recent.lock().unwrap_or_else(|e| e.into_inner());
                        if recent.len() == RECENT {
                            recent.pop_front();
                        }
                        recent.push_back(ev);
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                }
                me.touch();
            }
        })
    }

    async fn run(&self, command: Command) -> Result<ServerMsg, RemoteError> {
        let sup = self.host.supervisor();
        let done = match command {
            Command::Record => return Ok(ServerMsg::Record(self.record()?)),
            Command::Health => return Ok(ServerMsg::Health(self.health())),
            Command::Load { model, force } => {
                if self.host.draining() {
                    return Err(RemoteError::unavailable("this node is draining"));
                }
                sup.load_with(&model, force).await
            }
            Command::Unload { model } => sup.unload(&model).await,
            Command::Drain { on } => {
                self.host.set_draining(on);
                self.touch();
                if on {
                    let deadline = Instant::now() + self.drain_timeout;
                    while sup.in_flight() > 0 || self.admission.free() < self.slots {
                        if Instant::now() >= deadline {
                            return Err(RemoteError::Timeout);
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
                Ok(())
            }
            Command::Reserve { holder } => sup.reserve(&holder).await.map(|_| ()),
            Command::Release { holder } => sup.release(&holder).map(|_| ()),
            Command::GroupUp { plan } => {
                if let Some(r) = sup.reservation() {
                    return Err(RemoteError::unavailable(format!("this node is reserved by {}", r.holder)));
                }
                let workers = self.workers.as_ref().ok_or_else(|| RemoteError::Unsupported { message: "this node takes no group jobs".into() })?;
                for model in &plan.drop {
                    sup.unload(model).await?;
                }
                workers.up(&self.peer, &plan).await?;
                let mut groups = self.groups.lock().unwrap_or_else(|e| e.into_inner());
                if !groups.contains(&plan.group) {
                    groups.push(plan.group.clone());
                }
                Ok(())
            }
            Command::GroupDown { group } => {
                let workers = self.workers.as_ref().ok_or_else(|| RemoteError::Unsupported { message: "this node takes no group jobs".into() })?;
                workers.down(&group).await?;
                self.groups.lock().unwrap_or_else(|e| e.into_inner()).retain(|g| *g != group);
                Ok(())
            }
            Command::Subscribe => unreachable!("handled by serve"),
        };
        self.touch();
        done.map(|()| ServerMsg::Done)
    }

    /// Serves one node/1 stream. `caller` is the transport-authenticated peer.
    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(msg) = ch.recv_cbor::<ClientMsg>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), msg.cert) {
            (false, Some(cert)) => caller.with_cert(cert),
            (_, _) => caller,
        };
        if let Err(e) = self.auth.authorize(&caller) {
            return ch.send_cbor(&ServerMsg::Failed(e)).await;
        }
        if msg.command != Command::Subscribe {
            let answer = self.run(msg.command).await.unwrap_or_else(ServerMsg::Failed);
            return ch.send_cbor(&answer).await;
        }
        let mut changed = self.changed.subscribe();
        loop {
            if self.withdrawn.load(Ordering::SeqCst) {
                let seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
                return ch.send_cbor(&ServerMsg::Withdrawn { peer: self.peer.clone(), seq }).await;
            }
            ch.send_cbor(&ServerMsg::Record(self.record()?)).await?;
            changed.mark_unchanged();
            tokio::select! {
                r = changed.changed() => if r.is_err() { return Ok(()) },
                _ = tokio::time::sleep(REFRESH) => {}
                r = ch.rx.recv() => match r {
                    Ok(None) | Err(_) => return Ok(()),
                    Ok(Some(_)) => return Err(RemoteError::protocol("unexpected message on a subscription")),
                },
            }
        }
    }
}

/// How a [`NodeServer`] serves raw inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NodeServerOptions {
    /// Also serve raw inference on `/weirdnet-agent/infer/2` (and infer/1), for inference nodes
    /// on backends 0.1.0. Default on. Turn it off on a node that is also an inference node, so
    /// its inference role can serve infer/2 to apps.
    pub legacy_infer2: bool,
}

impl Default for NodeServerOptions {
    fn default() -> Self {
        Self { legacy_infer2: true }
    }
}

impl NodeServerOptions {
    pub fn with_legacy_infer2(mut self, on: bool) -> Self {
        self.legacy_infer2 = on;
        self
    }
}

/// Serves a [`NodeService`] on a `weirdnet-net` node: node/1, raw inference through the
/// service's [`ModelHost`] on [`RAW_INFER_PROTOCOL`] (and infer/2, see [`NodeServerOptions`];
/// no catalog, no inference announcement) and the [`MODEL_NODE_APP`] announcement, which lists
/// the protocols served. Dropping it stops serving.
pub struct NodeServer {
    _infer: Option<Server>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    stop_app: Box<dyn Fn() + Send + Sync>,
}

impl NodeServer {
    /// Starts serving with the default [`NodeServerOptions`] (raw inference on both ids). Call
    /// from a Tokio runtime. The service's key must be the node's.
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<NodeService>) -> Result<NodeServer, RemoteError> {
        Self::start_with(node, service, NodeServerOptions::default())
    }

    /// Starts serving with `options`.
    pub fn start_with<Q: Message, R: Message>(
        node: &Arc<Node<Q, R>>,
        service: Arc<NodeService>,
        options: NodeServerOptions,
    ) -> Result<NodeServer, RemoteError> {
        let net = |e: weirdnet_net::NetError| RemoteError::Failed { message: e.to_string() };
        if node.peer_id().to_string() != service.peer {
            return Err(RemoteError::Failed { message: "the service's keypair is not this node's".into() });
        }
        let host: Arc<dyn InferHost> = service.host.clone();
        let mut raw = node.accept_streams(RAW_INFER_PROTOCOL, stream_limits()).map_err(net)?;
        let (h, a) = (host.clone(), service.admission.clone());
        let raw_accept = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut raw).await {
                let caller = Caller::new(stream.peer().to_string());
                let (h, a) = (h.clone(), a.clone());
                tokio::spawn(async move {
                    let _ = weirdnet_agent::remote::infer::v2::serve(channel(stream), caller, h.as_ref(), &a).await;
                });
            }
        });
        let infer = if options.legacy_infer2 {
            Some(Server::start(node, Serving { infer: Some((host, service.admission.clone())), catalog: None, announce: false })?)
        } else {
            None
        };
        let mut incoming = node.accept_streams(PROTOCOL, stream_limits()).map_err(net)?;
        let svc = service.clone();
        let accept = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut incoming).await {
                let caller = Caller::new(stream.peer().to_string());
                let svc = svc.clone();
                tokio::spawn(async move {
                    let _ = svc.serve(channel(stream), caller).await;
                });
            }
        });
        let mut protocols = vec![PROTOCOL, RAW_INFER_PROTOCOL];
        if options.legacy_infer2 {
            protocols.push(weirdnet_agent::remote::infer::v2::PROTOCOL);
        }
        let body = to_cbor(&protocols)?;
        node.serve_app(
            MODEL_NODE_APP,
            Arc::new(move |_peer, _req| {
                let body = body.clone();
                async move { Ok(AppResponse::new(200).header("content-type", "application/cbor").body(AppBody::bytes(body))) }
            }),
        )
        .map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        let n = node.clone();
        Ok(NodeServer {
            _infer: infer,
            tasks: vec![accept, raw_accept, service.watch()],
            stop_app: Box::new(move || {
                n.stop_serving_app(MODEL_NODE_APP);
            }),
        })
    }
}

impl Drop for NodeServer {
    fn drop(&mut self) {
        for t in &self.tasks {
            t.abort();
        }
        (self.stop_app)();
    }
}

/// An update on a [`Subscription`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Update {
    Record(Box<Capability>),
    /// The node left; forget its records.
    Withdrawn,
}

/// Verified records from one model node.
pub struct Subscription {
    ch: Channel,
    node: String,
    verifier: Verifier,
    last_seq: u64,
    done: bool,
}

impl Subscription {
    /// The next update; `None` when the stream ended. Records are verified and their sequence
    /// numbers must increase.
    pub async fn next(&mut self) -> Option<Result<Update, RemoteError>> {
        if self.done {
            return None;
        }
        let msg = match self.ch.recv_cbor::<ServerMsg>().await {
            Ok(Some(m)) => m,
            Ok(None) => {
                self.done = true;
                return None;
            }
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let out = match msg {
            ServerMsg::Record(r) => self.verifier.verify(&r, &self.node, now_ms()).and_then(|cap| {
                if cap.seq <= self.last_seq {
                    return Err(RemoteError::protocol("capability record: sequence went backwards"));
                }
                self.last_seq = cap.seq;
                Ok(Update::Record(Box::new(cap)))
            }),
            ServerMsg::Withdrawn { seq, .. } if seq > self.last_seq => Ok(Update::Withdrawn),
            ServerMsg::Failed(e) => Err(e),
            other => Err(RemoteError::protocol(format!("unexpected {other:?} on a subscription"))),
        };
        if !matches!(out, Ok(Update::Record(_))) {
            self.done = true;
        }
        Some(out)
    }
}

/// An inference node's side of node/1: drives model nodes over a [`Dialer`]
/// (`weirdnet_agent::pool::NetDialer` on Weirdnet).
pub struct NodeClient {
    dialer: Arc<dyn Dialer>,
    verifier: Verifier,
    cert: Option<Vec<u8>>,
}

impl NodeClient {
    pub fn new(dialer: Arc<dyn Dialer>, verifier: Verifier) -> Self {
        Self { dialer, verifier, cert: None }
    }

    /// This node's CBOR `NodeCert` (`weirdnet_agent::pool::encode_cert`), presented to model nodes.
    pub fn with_cert(mut self, cert: Option<Vec<u8>>) -> Self {
        self.cert = cert;
        self
    }

    async fn open(&self, node: &str, command: Command) -> Result<Channel, RemoteError> {
        let mut ch = self.dialer.open(node, PROTOCOL).await?;
        ch.send_cbor(&ClientMsg { cert: self.cert.clone(), command }).await?;
        Ok(ch)
    }

    async fn call(&self, node: &str, command: Command) -> Result<ServerMsg, RemoteError> {
        let mut ch = self.open(node, command).await?;
        match ch.recv_cbor::<ServerMsg>().await? {
            Some(ServerMsg::Failed(e)) => Err(e),
            Some(m) => Ok(m),
            None => Err(RemoteError::Closed),
        }
    }

    async fn done(&self, node: &str, command: Command) -> Result<(), RemoteError> {
        match self.call(node, command).await? {
            ServerMsg::Done => Ok(()),
            other => Err(RemoteError::protocol(format!("expected done, got {other:?}"))),
        }
    }

    /// The node's current record, verified.
    pub async fn record(&self, node: &str) -> Result<Capability, RemoteError> {
        match self.call(node, Command::Record).await? {
            ServerMsg::Record(r) => self.verifier.verify(&r, node, now_ms()),
            other => Err(RemoteError::protocol(format!("expected a record, got {other:?}"))),
        }
    }

    /// Records now and on every change.
    pub async fn subscribe(&self, node: &str) -> Result<Subscription, RemoteError> {
        let ch = self.open(node, Command::Subscribe).await?;
        Ok(Subscription { ch, node: node.into(), verifier: self.verifier.clone(), last_seq: 0, done: false })
    }

    /// Loads `model` and waits until it is ready. `force` also loads a model in a crash loop.
    pub async fn load(&self, node: &str, model: &str, force: bool) -> Result<(), RemoteError> {
        self.done(node, Command::Load { model: model.into(), force }).await
    }

    pub async fn unload(&self, node: &str, model: &str) -> Result<(), RemoteError> {
        self.done(node, Command::Unload { model: model.into() }).await
    }

    /// Drains (waiting for requests under way) or resumes.
    pub async fn drain(&self, node: &str, on: bool) -> Result<(), RemoteError> {
        self.done(node, Command::Drain { on }).await
    }

    /// Reserves the node's GPU for `holder` ([`Command::Reserve`]): returns once its engines are
    /// stopped. A node before 0.4.3 fails the call.
    pub async fn reserve(&self, node: &str, holder: &str) -> Result<(), RemoteError> {
        self.done(node, Command::Reserve { holder: holder.into() }).await
    }

    /// Ends `holder`'s reservation.
    pub async fn release(&self, node: &str, holder: &str) -> Result<(), RemoteError> {
        self.done(node, Command::Release { holder: holder.into() }).await
    }

    pub async fn group_up(&self, node: &str, plan: GroupPlan) -> Result<(), RemoteError> {
        self.done(node, Command::GroupUp { plan }).await
    }

    pub async fn group_down(&self, node: &str, group: &str) -> Result<(), RemoteError> {
        self.done(node, Command::GroupDown { group: group.into() }).await
    }

    pub async fn health(&self, node: &str) -> Result<Health, RemoteError> {
        match self.call(node, Command::Health).await? {
            ServerMsg::Health(h) => Ok(h),
            other => Err(RemoteError::protocol(format!("expected health, got {other:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{Supervisor, SupervisorOptions};
    use weirdnet_net::WEIRDNET;

    fn service() -> (NodeService, Keypair) {
        let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
        let auth: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let host = Arc::new(ModelHost::new(sup, auth.clone()));
        let kp = Keypair::generate_ed25519();
        (NodeService::new(host, kp.clone(), None, auth), kp)
    }

    #[test]
    fn records_verify_and_tampering_fails() {
        let (svc, _) = service();
        let rec = svc.record().unwrap();
        let v = Verifier::signature_only(&WEIRDNET);
        let cap = v.verify(&rec, svc.peer(), now_ms()).unwrap();
        assert_eq!((cap.peer.as_str(), cap.draining, cap.group_jobs), (svc.peer(), false, false));
        assert!(cap.expires_ms - cap.issued_ms == 90_000);
        let mut bad = rec.clone();
        let mid = bad.record.len() / 2;
        bad.record[mid] ^= 1;
        assert!(v.verify(&bad, svc.peer(), now_ms()).is_err());
        assert!(v.verify(&rec, &PeerId::random().to_string(), now_ms()).is_err(), "another peer's record");
        assert!(v.verify(&rec, svc.peer(), cap.expires_ms).is_err(), "expired");
        let user = Keypair::generate_ed25519();
        assert!(Verifier::new(&WEIRDNET, user.public()).verify(&rec, svc.peer(), now_ms()).is_err(), "no certificate");
        assert!(svc.record().unwrap().verify(svc.peer(), now_ms()).unwrap().seq > cap.seq);
    }
}
