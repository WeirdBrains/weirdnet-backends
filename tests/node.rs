//! `/weirdnet-backends/node/1` and raw infer/2 between in-process `weirdnet-net` nodes (memory
//! transport): an inference node driving a model node of the same user, and a stranger.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use weirdnet_agent::pool::{NetDialer, Pool, PoolOptions, encode_cert};
use weirdnet_agent::remote::infer::v2::{Finish, InferBackend, Output, Request};
use weirdnet_agent::remote::infer::{Admission, InferHost};
use weirdnet_agent::remote::pool::NoEvents;
use weirdnet_agent::remote::{Authorize, Caller, RemoteError};
use weirdnet_agent::serve::RosterAuth;
use weirdnet_agent::serve::{Server, Serving};
use weirdnet_agent::{CancelToken, Message, ModelBackend};
use weirdnet_backends::node::{
    AddrClass, Capability, GroupPlan, GroupWorkers, LinkInfo, NodeClient, NodeServer, NodeServerOptions, NodeService, Subscription, Update, Verifier,
};
use weirdnet_backends::{EngineKind, ModelHost, ModelState, RawInferDialer};
use weirdnet_net::node::{Handler, Node, NodeConfig};
use weirdnet_net::user::NodeCert;
use weirdnet_net::{Keypair, Multiaddr, PeerId, WEIRDNET};
use weirdnet_sync::Roster;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Q;
#[derive(Debug, Clone, Serialize, Deserialize)]
struct R;
struct H;
impl Handler<Q, R> for H {
    fn request(&self, _: &PeerId, _: Q) -> R {
        R
    }
}
type N = Node<Q, R>;

fn start(kp: Keypair) -> Arc<N> {
    let cfg = NodeConfig { listen: vec!["/memory/0".parse().unwrap()], memory: true, ..Default::default() };
    Arc::new(Node::start(&WEIRDNET, kp, cfg, Arc::new(H)).unwrap())
}

fn addr(n: &N) -> Multiaddr {
    n.reachable_addrs().into_iter().find(|a| a.starts_with("/memory/") && !a.contains("p2p-circuit")).unwrap().parse().unwrap()
}

fn certify(user: &Keypair, n: &N, label: &str) -> NodeCert {
    Roster::self_certify(&WEIRDNET, user, &n.peer_id().to_string(), label, weirdnet_net::now_ms()).unwrap()
}

#[derive(Default)]
struct Workers {
    calls: Mutex<Vec<String>>,
}

#[async_trait]
impl GroupWorkers for Workers {
    async fn up(&self, me: &str, plan: &GroupPlan) -> Result<(), RemoteError> {
        self.calls.lock().unwrap().push(format!("up {} {} leader={}", plan.group, me == plan.leader, plan.worker.as_deref().unwrap_or("")));
        Ok(())
    }
    async fn down(&self, group: &str) -> Result<(), RemoteError> {
        self.calls.lock().unwrap().push(format!("down {group}"));
        Ok(())
    }
}

/// The next record satisfying `want`.
async fn until(sub: &mut Subscription, want: impl Fn(&Capability) -> bool) -> Capability {
    let wait = async {
        loop {
            match sub.next().await {
                Some(Ok(Update::Record(cap))) if want(&cap) => return *cap,
                Some(Ok(Update::Record(_))) => {}
                other => panic!("subscription: {other:?}"),
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(10), wait).await.expect("no matching record")
}

fn state(cap: &Capability, id: &str) -> ModelState {
    cap.models.iter().find(|m| m.status.id == id).unwrap().status.state.clone()
}

fn node1_round_trip() {
    let user = Keypair::generate_ed25519();
    let model_key = Keypair::generate_ed25519();
    let (inference, model, stranger) = (start(Keypair::generate_ed25519()), start(model_key.clone()), start(Keypair::generate_ed25519()));
    let inference_cert = certify(&user, &inference, "studio");
    let model_cert = certify(&user, &model, "gpu-box");
    let mut roster = Roster::new();
    roster.upsert(inference_cert.clone());
    roster.upsert(model_cert.clone());
    let roster = Arc::new(roster);
    let view: Arc<dyn Fn() -> Roster + Send + Sync> = Arc::new(move || (*roster).clone());

    common::block_on(async {
        let auth: Arc<dyn Authorize> = Arc::new(RosterAuth::new(&WEIRDNET, user.public(), view.clone()));
        let sup = common::supervisor(vec![common::launch("chat", EngineKind::Llamacpp)]);
        let host = Arc::new(ModelHost::new(sup.clone(), auth.clone()));
        let workers = Arc::new(Workers::default());
        let service = Arc::new(NodeService::new(host, model_key, Some(model_cert), auth).with_admission(2, 2).with_group_workers(workers.clone()));
        service.set_model_info("chat", Some("abc123".into()), Some("Q4_K_M".into()));
        let _server = NodeServer::start(&model, service.clone()).unwrap();
        let peer = model.peer_id().to_string();

        let dialer = Arc::new(NetDialer::new(inference.clone()));
        dialer.hint(model.peer_id(), vec![addr(&model)]);
        let client = NodeClient::new(dialer.clone(), Verifier::new(&WEIRDNET, user.public())).with_cert(Some(encode_cert(&inference_cert)));

        // Signed records, chained to the user.
        let cap = client.record(&peer).await.unwrap();
        assert_eq!((cap.peer.as_str(), cap.group_jobs, cap.draining), (peer.as_str(), true, false));
        assert_eq!(state(&cap, "chat"), ModelState::Unloaded);
        let entry = cap.models.iter().find(|m| m.status.id == "chat").unwrap();
        assert_eq!((entry.revision.as_deref(), entry.quant.as_deref()), (Some("abc123"), Some("Q4_K_M")));
        assert_eq!(cap.engines, vec![EngineKind::Llamacpp]);
        assert_eq!(cap.load.slots, 2);
        let mut sub = client.subscribe(&peer).await.unwrap();
        until(&mut sub, |c| state(c, "chat") == ModelState::Unloaded).await;

        // Load: pushed as a new record.
        client.load(&peer, "chat", false).await.unwrap();
        until(&mut sub, |c| state(c, "chat") == ModelState::Ready).await;

        // Raw infer/2 from the inference node.
        let pool = Pool::new(dialer.clone(), vec![peer.clone()]).with_options(PoolOptions::default().with_cert(Some(encode_cert(&inference_cert))));
        let cancel = CancelToken::new();
        let answer = pool.infer(Request::messages(vec![Message::user("hi")]).with_model("chat").raw(), &NoEvents, &cancel).await.unwrap();
        assert_eq!(answer.text, "Hello from chat! You said: hi");

        // Health.
        let health = client.health(&peer).await.unwrap();
        assert!(health.ok && !health.draining);
        assert!(health.events.iter().any(|e| matches!(e, weirdnet_backends::SupervisorEvent::Ready { .. })), "{:?}", health.events);

        // Groups go through the application's workers; `drop` models unload first.
        let plan = GroupPlan::new("big", peer.clone()).with_drop(vec!["chat".into()]).with_worker("rpc-server").with_link("10.0.0.2:50052");
        client.group_up(&peer, plan).await.unwrap();
        let cap = until(&mut sub, |c| c.groups == ["big"]).await;
        assert_eq!(state(&cap, "chat"), ModelState::Unloaded);
        client.group_down(&peer, "big").await.unwrap();
        until(&mut sub, |c| c.groups.is_empty()).await;
        assert_eq!(*workers.calls.lock().unwrap(), ["up big true leader=rpc-server", "down big"]);

        // Drain: no new requests or loads; then back.
        client.load(&peer, "chat", false).await.unwrap();
        client.drain(&peer, true).await.unwrap();
        until(&mut sub, |c| c.draining).await;
        let err = pool.infer(Request::messages(vec![Message::user("hi")]).with_model("chat").raw(), &NoEvents, &cancel).await.unwrap_err();
        assert!(matches!(err.error, RemoteError::Unavailable { .. }), "{:?}", err.error);
        assert!(matches!(client.load(&peer, "chat", false).await, Err(RemoteError::Unavailable { .. })));
        client.drain(&peer, false).await.unwrap();
        until(&mut sub, |c| !c.draining).await;

        // A stranger with another user's certificate is refused.
        let foreign = Keypair::generate_ed25519();
        let fake = Roster::self_certify(&WEIRDNET, &foreign, &stranger.peer_id().to_string(), "x", weirdnet_net::now_ms()).unwrap();
        let sd = Arc::new(NetDialer::new(stranger.clone()));
        sd.hint(model.peer_id(), vec![addr(&model)]);
        let outsider = NodeClient::new(sd, Verifier::signature_only(&WEIRDNET)).with_cert(Some(encode_cert(&fake)));
        assert!(matches!(outsider.unload(&peer, "chat").await, Err(RemoteError::Refused { .. })));
        assert!(matches!(outsider.record(&peer).await, Err(RemoteError::Refused { .. })));

        // A record from a node the user did not certify does not verify.
        let other = Verifier::new(&WEIRDNET, foreign.public());
        let honest = service.record().unwrap();
        assert!(matches!(other.verify(&honest, &peer, weirdnet_net::now_ms()), Err(RemoteError::Refused { .. })));

        // Private links go into the record; public and CGNAT addresses never do.
        let links = ["10.0.0.2", "8.8.8.8", "100.100.1.1"].map(|a| LinkInfo::new(a.parse().unwrap()).with_declared_gbps(Some(25.0)));
        service.set_links(links.to_vec());
        let cap = until(&mut sub, |c| !c.links.is_empty()).await;
        assert_eq!(cap.links.iter().map(|l| l.addr.as_str()).collect::<Vec<_>>(), ["10.0.0.2"]);
        assert_eq!((cap.links[0].class, cap.links[0].best_gbps()), (AddrClass::Private, Some(25.0)));

        // With the roster, a revoked model node's record is refused although its cert is valid.
        let linked = Verifier::new(&WEIRDNET, user.public()).with_roster(view.clone());
        linked.verify(&honest, &peer, weirdnet_net::now_ms()).unwrap();
        let mut after = view();
        after.revoke(&WEIRDNET, &user, &peer, weirdnet_net::now_ms()).unwrap();
        let revoked = Verifier::new(&WEIRDNET, user.public()).with_roster(Arc::new(move || after.clone()));
        let err = revoked.verify(&honest, &peer, weirdnet_net::now_ms()).unwrap_err();
        assert!(matches!(&err, RemoteError::Refused { message } if message.contains("revoked")), "{err:?}");
        Verifier::new(&WEIRDNET, user.public()).verify(&honest, &peer, weirdnet_net::now_ms()).unwrap();
        let empty = Verifier::new(&WEIRDNET, user.public()).with_roster(Arc::new(Roster::new));
        assert!(empty.verify(&honest, &peer, weirdnet_net::now_ms()).is_err(), "unknown to the roster");

        // Withdrawal ends subscriptions.
        service.withdraw();
        let end = async {
            loop {
                match sub.next().await {
                    Some(Ok(Update::Record(_))) => {}
                    Some(Ok(Update::Withdrawn)) => return,
                    other => panic!("{other:?}"),
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(10), end).await.unwrap();
        sup.shutdown().await;
    });
}

/// The inference role's own answer (standing in for an inference node's routing).
struct Router;

#[async_trait]
impl InferBackend for Router {
    fn name(&self) -> String {
        "router".into()
    }
    async fn infer(&self, _request: &Request, out: &Output, _cancel: &CancelToken) -> Result<Finish, RemoteError> {
        out.text("from the inference role");
        Ok(Finish::new("stop"))
    }
}

struct RouterHost(Arc<dyn Authorize>);

impl InferHost for RouterHost {
    fn authorize(&self, caller: &Caller) -> Result<(), RemoteError> {
        self.0.authorize(caller)
    }
    fn open(&self, _caller: &Caller, _model: Option<&str>) -> Result<Arc<dyn ModelBackend>, RemoteError> {
        Err(RemoteError::Unsupported { message: "infer/2 only".into() })
    }
    fn open_request(&self, _caller: &Caller, _request: &Request) -> Result<Arc<dyn InferBackend>, RemoteError> {
        Ok(Arc::new(Router))
    }
}

/// One node is both an inference node (infer/2 for apps) and a model node (raw inference on
/// /weirdnet-backends/infer/1, legacy infer/2 off); another app's device reaches each role.
fn one_node_is_inference_and_model_node() {
    let user = Keypair::generate_ed25519();
    let both_key = Keypair::generate_ed25519();
    let (both, app) = (start(both_key.clone()), start(Keypair::generate_ed25519()));
    let both_cert = certify(&user, &both, "studio");
    let app_cert = certify(&user, &app, "laptop");
    let mut roster = Roster::new();
    roster.upsert(both_cert.clone());
    roster.upsert(app_cert.clone());
    let roster = Arc::new(roster);
    let view: Arc<dyn Fn() -> Roster + Send + Sync> = Arc::new(move || (*roster).clone());

    common::block_on(async {
        let auth: Arc<dyn Authorize> = Arc::new(RosterAuth::new(&WEIRDNET, user.public(), view.clone()));
        let sup = common::supervisor(vec![common::launch("chat", EngineKind::Llamacpp)]);
        sup.load("chat").await.unwrap();
        let host = Arc::new(ModelHost::new(sup.clone(), auth.clone()).with_default("chat"));
        let service = Arc::new(NodeService::new(host, both_key, Some(both_cert), auth.clone()));
        let _model = NodeServer::start_with(&both, service, NodeServerOptions::default().with_legacy_infer2(false)).unwrap();
        let router: Arc<dyn InferHost> = Arc::new(RouterHost(auth));
        let _inference = Server::start(&both, Serving { infer: Some((router, Admission::new(2))), catalog: None, announce: true }).unwrap();
        let peer = both.peer_id().to_string();
        let opts = PoolOptions::default().with_cert(Some(encode_cert(&app_cert)));
        let cancel = CancelToken::new();

        let dialer = Arc::new(NetDialer::new(app.clone()));
        dialer.hint(both.peer_id(), vec![addr(&both)]);
        // An app: plain infer/2 reaches the inference role.
        let apps = Pool::new(dialer.clone(), vec![peer.clone()]).with_options(opts.clone());
        let answer = apps.infer(Request::messages(vec![Message::user("hi")]), &NoEvents, &cancel).await.unwrap();
        assert_eq!(answer.text, "from the inference role");
        // An inference node driving model nodes: the dialer goes to /weirdnet-backends/infer/1.
        let raw = Arc::new(RawInferDialer::new(dialer.clone()));
        let models = Pool::new(raw.clone(), vec![peer.clone()]).with_options(opts);
        let answer = models.infer(Request::messages(vec![Message::user("hi")]).raw(), &NoEvents, &cancel).await.unwrap();
        assert_eq!(answer.text, "Hello from chat! You said: hi");
        assert!(!raw.is_legacy(&peer));
        sup.shutdown().await;
    });
}

fn main() {
    common::main(&[
        ("node1_round_trip", node1_round_trip as fn()),
        ("one_node_is_inference_and_model_node", one_node_is_inference_and_model_node as fn()),
    ]);
}
