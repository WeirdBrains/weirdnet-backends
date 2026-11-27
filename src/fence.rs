//! Fencing for placement leases on the model node (since 0.4.11).
//!
//! A scheduler that holds a lease (a `weirdnet_sync::Leases` lease, for example an inference
//! node's placement lease) attaches the lease name and its term, a [`Fence`], to node/1 `Load`
//! and `Unload` (`Command::LoadFenced` and `Command::UnloadFenced` with the `node` feature). The
//! model node keeps a [`FenceGuard`]: for each lease name, the highest term it has seen. A command
//! with an older term is refused ([`FenceRefusal::Stale`]) and changes nothing, so a holder that
//! lost the lease cannot load or unload models once a newer holder has reached the node.
//!
//! The rule is `weirdnet_sync::FenceGuard`'s: one counter per lease name, a term equal to the
//! highest seen passes, an older one is stale. The persisted bytes are the same too:
//! [`FenceGuard::export`] and `weirdnet_sync::FenceGuard::export` read each other's output. This
//! crate keeps its own copy so it does not require a newer `weirdnet-sync` than its callers use.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use weirdnet_agent::remote::{RemoteError, from_cbor, to_cbor};

const MAX_NAME: usize = 200;
/// Message prefix of a stale-term refusal ([`FenceRefusal::Stale`]).
const STALE: &str = "stale fence: ";
/// Message prefix of a refused unfenced command ([`FenceRefusal::Required`]).
const REQUIRED: &str = "fence required: ";

/// A lease name and term, attached to a command by the lease holder.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Fence {
    /// The lease name, as `weirdnet_sync::LeaseName`: 1 to 200 of `A-Z a-z 0-9 . - _ : / + #`.
    pub lease: String,
    /// The holder's term (`weirdnet_sync::Term`): strictly increases per lease name across
    /// holders.
    pub term: u64,
}

impl Fence {
    /// `lease` and `term` from the holder's lease handle (`handle.name().as_str()` and
    /// `handle.term().0` with `weirdnet-sync`).
    pub fn new(lease: impl Into<String>, term: u64) -> Self {
        Self { lease: lease.into(), term }
    }

    /// Whether the lease name is one `weirdnet_sync::LeaseName` accepts.
    pub fn is_valid(&self) -> bool {
        valid_lease_name(&self.lease)
    }
}

impl std::fmt::Display for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} term {}", self.lease, self.term)
    }
}

/// Whether `name` is a lease name `weirdnet_sync::LeaseName` accepts: 1 to 200 of
/// `A-Z a-z 0-9 . - _ : / + #`.
pub fn valid_lease_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= MAX_NAME && name.chars().all(|c| c.is_ascii_alphanumeric() || ".-_:/+#".contains(c))
}

/// The resource side of fencing: refuses a command that carries an older term than one already
/// seen for its lease name. Persist it ([`save`](Self::save) or [`export`](Self::export)) so a
/// restart keeps refusing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FenceGuard {
    seen: HashMap<String, u64>,
}

/// A command from an older term than the guard has seen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stale {
    /// The highest term seen for the lease name.
    pub seen: u64,
}

impl std::fmt::Display for Stale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stale term: {} already seen", self.seen)
    }
}

impl std::error::Error for Stale {}

impl FenceGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accepts `term` if it is at least the highest seen for `lease`, and records it.
    pub fn check(&mut self, lease: &str, term: u64) -> Result<(), Stale> {
        let seen = self.seen.entry(lease.to_string()).or_default();
        if term < *seen {
            return Err(Stale { seen: *seen });
        }
        *seen = term;
        Ok(())
    }

    /// The highest term seen for `lease`.
    pub fn seen(&self, lease: &str) -> Option<u64> {
        self.seen.get(lease).copied()
    }

    /// Every lease name seen, with its highest term, sorted by name.
    pub fn leases(&self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self.seen.iter().map(|(n, t)| (n.clone(), *t)).collect();
        v.sort();
        v
    }

    /// CBOR, the same bytes as `weirdnet_sync::FenceGuard::export`.
    pub fn export(&self) -> Vec<u8> {
        to_cbor(&self.leases()).unwrap_or_default()
    }

    /// Reads [`export`](Self::export)'s or `weirdnet_sync::FenceGuard::export`'s bytes. Bad lease
    /// names are refused.
    pub fn import(bytes: &[u8]) -> Result<Self, RemoteError> {
        let v: Vec<(String, u64)> = from_cbor(bytes)?;
        if let Some((bad, _)) = v.iter().find(|(n, _)| !valid_lease_name(n)) {
            return Err(RemoteError::protocol(format!("fence guard: bad lease name {bad:?}")));
        }
        Ok(Self { seen: v.into_iter().collect() })
    }

    /// The guard saved at `path`, or an empty one when there is no file yet.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        match std::fs::read(path.as_ref()) {
            Ok(bytes) => Self::import(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::new()),
            Err(e) => Err(e),
        }
    }

    /// Saves to `path`: written to a temporary file next to it, synced, then renamed over it.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = std::path::PathBuf::from(tmp);
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&self.export())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }
}

/// Why a model node refused a command for fencing. It travels as a [`RemoteError::Refused`]
/// with a fixed message; [`from_error`](Self::from_error) reads it back, so a caller can tell a
/// fencing refusal (stop: another holder has the lease, or the node wants a fence) from a
/// transient failure (retry).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FenceRefusal {
    /// The node has seen a newer term for the lease; the command changed nothing.
    Stale { lease: String, term: u64, seen: u64 },
    /// The node takes `Load` and `Unload` only with a fence, and the command had none.
    Required,
}

impl FenceRefusal {
    /// The error sent to the caller: `Refused` with `stale fence: lease=NAME term=N seen=M` or
    /// `fence required: ...`.
    pub fn to_error(&self) -> RemoteError {
        match self {
            FenceRefusal::Stale { lease, term, seen } => RemoteError::refused(format!("{STALE}lease={lease} term={term} seen={seen}")),
            FenceRefusal::Required => RemoteError::refused(format!("{REQUIRED}this node takes Load and Unload only with a lease fence")),
        }
    }

    /// The refusal in `error`, if it is one.
    pub fn from_error(error: &RemoteError) -> Option<Self> {
        let RemoteError::Refused { message } = error else { return None };
        if message.starts_with(REQUIRED) {
            return Some(FenceRefusal::Required);
        }
        let rest = message.strip_prefix(STALE)?;
        let (mut lease, mut term, mut seen) = (None, None, None);
        for part in rest.split_whitespace() {
            match part.split_once('=') {
                Some(("lease", v)) => lease = Some(v.to_string()),
                Some(("term", v)) => term = v.parse().ok(),
                Some(("seen", v)) => seen = v.parse().ok(),
                _ => {}
            }
        }
        Some(FenceRefusal::Stale { lease: lease?, term: term?, seen: seen? })
    }
}

impl std::fmt::Display for FenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FenceRefusal::Stale { lease, term, seen } => write!(f, "stale fence: lease {lease} term {term}, term {seen} already seen"),
            FenceRefusal::Required => f.write_str("fence required"),
        }
    }
}

impl std::error::Error for FenceRefusal {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_only_go_up_per_lease() {
        let mut g = FenceGuard::new();
        assert_eq!(g.seen("ymir.placement"), None);
        g.check("ymir.placement", 1).unwrap();
        g.check("ymir.placement", 1).unwrap();
        g.check("ymir.placement", 2).unwrap();
        assert_eq!(g.check("ymir.placement", 1), Err(Stale { seen: 2 }));
        assert_eq!(g.seen("ymir.placement"), Some(2), "a refused term changes nothing");
        // Another lease name has its own counter.
        g.check("ymir.jobs/7", 1).unwrap();
        assert_eq!((g.seen("ymir.placement"), g.seen("ymir.jobs/7")), (Some(2), Some(1)));
        assert_eq!(g.leases(), vec![("ymir.jobs/7".to_string(), 1), ("ymir.placement".to_string(), 2)]);
    }

    #[test]
    fn saved_guards_survive_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state").join("fence.cbor");
        assert_eq!(FenceGuard::load(&path).unwrap(), FenceGuard::new(), "no file yet");
        let mut g = FenceGuard::new();
        g.check("ymir.placement", 5).unwrap();
        g.save(&path).unwrap();
        let mut back = FenceGuard::load(&path).unwrap();
        assert_eq!(back, g);
        assert_eq!(back.check("ymir.placement", 4), Err(Stale { seen: 5 }));
        assert!(!dir.path().join("state").join("fence.cbor.tmp").exists());
        std::fs::write(&path, b"not cbor").unwrap();
        assert_eq!(FenceGuard::load(&path).unwrap_err().kind(), std::io::ErrorKind::InvalidData);
        let bad = to_cbor(&vec![("bad name".to_string(), 1u64)]).unwrap();
        assert!(FenceGuard::import(&bad).is_err());
    }

    #[test]
    fn refusals_read_back_from_errors() {
        for r in [FenceRefusal::Stale { lease: "ymir.placement".into(), term: 1, seen: 2 }, FenceRefusal::Required] {
            let e = r.to_error();
            assert!(matches!(e, RemoteError::Refused { .. }));
            assert_eq!(FenceRefusal::from_error(&e), Some(r.clone()));
            // Through the wire too.
            let back: RemoteError = from_cbor(&to_cbor(&e).unwrap()).unwrap();
            assert_eq!(FenceRefusal::from_error(&back), Some(r));
        }
        assert_eq!(FenceRefusal::from_error(&RemoteError::refused("not a linked device")), None);
        assert_eq!(FenceRefusal::from_error(&RemoteError::unavailable("stale fence: lease=a term=1 seen=2")), None);
        assert_eq!(FenceRefusal::from_error(&RemoteError::Busy), None);
    }

    /// The bytes `weirdnet_sync::FenceGuard::export` (0.8) gives after the same checks; CI also
    /// tests against weirdnet-sync 0.5 and 0.7, which have no `FenceGuard`, so they are fixed here.
    const SYNC_EXPORT: [u8; 32] = [
        0x82, 0x82, 0x6b, b'y', b'm', b'i', b'r', b'.', b'j', b'o', b'b', b's', b'/', b'7', 0x02, 0x82, 0x6e, b'y', b'm', b'i', b'r', b'.', b'p',
        b'l', b'a', b'c', b'e', b'm', b'e', b'n', b't', 0x03,
    ];

    #[test]
    fn saved_bytes_match_weirdnet_sync() {
        let mut g = FenceGuard::new();
        // The same outcomes as weirdnet_sync::FenceGuard for these checks.
        let outcomes: Vec<_> = [("ymir.placement", 1), ("ymir.placement", 3), ("ymir.jobs/7", 2), ("ymir.placement", 2), ("ymir.jobs/7", 2)]
            .into_iter()
            .map(|(lease, term)| g.check(lease, term).map_err(|s| s.seen))
            .collect();
        assert_eq!(outcomes, vec![Ok(()), Ok(()), Ok(()), Err(3), Ok(())]);
        assert_eq!(g.export(), SYNC_EXPORT);
        assert_eq!(FenceGuard::import(&SYNC_EXPORT).unwrap(), g);
    }

    #[test]
    fn lease_names_follow_weirdnet_sync() {
        assert!(Fence::new("ymir.placement", 1).is_valid());
        assert!(Fence::new("ymir.jobs/a-1_b:c+d#e", 1).is_valid());
        assert!(!Fence::new("", 1).is_valid());
        assert!(!Fence::new("has space", 1).is_valid());
        assert!(!Fence::new("x".repeat(201), 1).is_valid());
        assert!(Fence::new("x".repeat(200), 1).is_valid());
    }
}
