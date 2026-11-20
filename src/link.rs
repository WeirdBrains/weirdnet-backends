//! Link info for capability records (M7 phase 1): a model node's private interfaces, so the
//! planner can place swap groups on machines with a fast private link.
//!
//! Only loopback, link-local and private (RFC 1918, unique local) addresses are ever advertised
//! ([`AddrClass::advertised`]); public and carrier-grade NAT addresses never leave the node.
//! Speeds are the NIC's nominal speed from the OS and an operator-declared figure; measured
//! figures (`measured`) are reserved for the later probe command.

use std::net::IpAddr;

use serde::{Deserialize, Serialize};

/// What kind of address an interface has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum AddrClass {
    Loopback,
    /// 169.254/16, fe80::/10.
    LinkLocal,
    /// RFC 1918, fc00::/7.
    Private,
    /// 100.64/10 (carrier-grade NAT, also Tailscale).
    Cgnat,
    Public,
}

impl AddrClass {
    pub fn of(ip: IpAddr) -> Self {
        match ip {
            IpAddr::V4(v4) => {
                let o = v4.octets();
                if v4.is_loopback() {
                    Self::Loopback
                } else if v4.is_link_local() {
                    Self::LinkLocal
                } else if v4.is_private() {
                    Self::Private
                } else if o[0] == 100 && (o[1] & 0xc0) == 64 {
                    Self::Cgnat
                } else {
                    Self::Public
                }
            }
            IpAddr::V6(v6) => {
                if let Some(v4) = v6.to_ipv4_mapped() {
                    return Self::of(IpAddr::V4(v4));
                }
                let s = v6.segments();
                if v6.is_loopback() {
                    Self::Loopback
                } else if (s[0] & 0xffc0) == 0xfe80 {
                    Self::LinkLocal
                } else if (s[0] & 0xfe00) == 0xfc00 {
                    Self::Private
                } else {
                    Self::Public
                }
            }
        }
    }

    /// May go into a capability record (not `Public`, not `Cgnat`).
    pub fn advertised(self) -> bool {
        matches!(self, Self::Loopback | Self::LinkLocal | Self::Private)
    }
}

/// One measured link to a peer (newest kept per peer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LinkMeasure {
    pub peer: String,
    pub gbps: f32,
    pub rtt_us: u32,
    pub at_ms: i64,
}

impl LinkMeasure {
    pub fn new(peer: impl Into<String>, gbps: f32, rtt_us: u32, at_ms: i64) -> Self {
        Self { peer: peer.into(), gbps, rtt_us, at_ms }
    }
}

/// A private interface of a model node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LinkInfo {
    /// The interface's IP address.
    pub addr: String,
    pub class: AddrClass,
    #[serde(default)]
    pub iface: Option<String>,
    /// NIC speed from the OS (Linux `/sys/class/net/*/speed`).
    #[serde(default)]
    pub nominal_gbps: Option<f32>,
    /// The operator's figure (config).
    #[serde(default)]
    pub declared_gbps: Option<f32>,
    #[serde(default)]
    pub measured: Vec<LinkMeasure>,
}

impl LinkInfo {
    /// The class comes from the address.
    pub fn new(addr: IpAddr) -> Self {
        Self { addr: addr.to_string(), class: AddrClass::of(addr), iface: None, nominal_gbps: None, declared_gbps: None, measured: Vec::new() }
    }

    pub fn with_iface(mut self, iface: impl Into<String>) -> Self {
        self.iface = Some(iface.into());
        self
    }

    pub fn with_nominal_gbps(mut self, gbps: Option<f32>) -> Self {
        self.nominal_gbps = gbps;
        self
    }

    pub fn with_declared_gbps(mut self, gbps: Option<f32>) -> Self {
        self.declared_gbps = gbps;
        self
    }

    /// The best figure known: measured (fastest recent), else declared, else nominal.
    pub fn best_gbps(&self) -> Option<f32> {
        self.measured.iter().map(|m| m.gbps).reduce(f32::max).or(self.declared_gbps).or(self.nominal_gbps)
    }
}

/// Keeps only advertisable links (drops public and CGNAT addresses) and caps the count.
pub fn advertised(links: impl IntoIterator<Item = LinkInfo>) -> Vec<LinkInfo> {
    links
        .into_iter()
        .filter(|l| l.class.advertised() && l.addr.parse::<IpAddr>().is_ok_and(|ip| AddrClass::of(ip) == l.class))
        .take(MAX_LINKS)
        .collect()
}

/// Most links a record carries.
pub const MAX_LINKS: usize = 16;

/// This machine's advertisable interfaces (no loopback), with the nominal speed where the OS
/// reports it. Empty where interfaces can't be listed (non-Unix).
pub fn local_links() -> Vec<LinkInfo> {
    let mut out: Vec<LinkInfo> = interfaces()
        .into_iter()
        .filter(|(_, ip)| AddrClass::of(*ip).advertised() && !ip.is_loopback())
        .map(|(iface, ip)| {
            let speed = nominal_gbps(&iface);
            LinkInfo::new(ip).with_iface(iface).with_nominal_gbps(speed)
        })
        .collect();
    out.sort_by(|a, b| (&a.iface, &a.addr).cmp(&(&b.iface, &b.addr)));
    out.dedup_by(|a, b| a.addr == b.addr);
    out.truncate(MAX_LINKS);
    out
}

#[cfg(target_os = "linux")]
fn nominal_gbps(iface: &str) -> Option<f32> {
    if iface.contains('/') || iface.contains("..") {
        return None;
    }
    let mbps: i64 = std::fs::read_to_string(format!("/sys/class/net/{iface}/speed")).ok()?.trim().parse().ok()?;
    (mbps > 0).then(|| mbps as f32 / 1000.0)
}

#[cfg(not(target_os = "linux"))]
fn nominal_gbps(_iface: &str) -> Option<f32> {
    None
}

#[cfg(unix)]
fn interfaces() -> Vec<(String, IpAddr)> {
    use std::ffi::CStr;
    use std::net::{Ipv4Addr, Ipv6Addr};
    let mut out = Vec::new();
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs fills `head` with a list we only read and then free once.
    if unsafe { libc::getifaddrs(&mut head) } != 0 {
        return out;
    }
    let mut cur = head;
    while !cur.is_null() {
        // SAFETY: `cur` is a node of the list getifaddrs returned, valid until freeifaddrs.
        let ifa = unsafe { &*cur };
        cur = ifa.ifa_next;
        if ifa.ifa_addr.is_null() || ifa.ifa_name.is_null() {
            continue;
        }
        // SAFETY: ifa_name is a NUL-terminated C string owned by the list.
        let name = unsafe { CStr::from_ptr(ifa.ifa_name) }.to_string_lossy().into_owned();
        // SAFETY: ifa_addr is non-null; the family says which sockaddr it is.
        let ip = match i32::from(unsafe { (*ifa.ifa_addr).sa_family }) {
            libc::AF_INET => {
                let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in) };
                IpAddr::V4(Ipv4Addr::from(u32::from_be(sa.sin_addr.s_addr)))
            }
            libc::AF_INET6 => {
                let sa = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in6) };
                IpAddr::V6(Ipv6Addr::from(sa.sin6_addr.s6_addr))
            }
            _ => continue,
        };
        out.push((name, ip));
    }
    // SAFETY: frees the list returned above, once.
    unsafe { libc::freeifaddrs(head) };
    out
}

#[cfg(not(unix))]
fn interfaces() -> Vec<(String, IpAddr)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn addresses_are_classed_and_only_private_ones_advertised() {
        for (a, c) in [
            ("127.0.0.1", AddrClass::Loopback),
            ("::1", AddrClass::Loopback),
            ("169.254.3.4", AddrClass::LinkLocal),
            ("fe80::1", AddrClass::LinkLocal),
            ("10.0.0.2", AddrClass::Private),
            ("172.16.5.5", AddrClass::Private),
            ("192.168.1.5", AddrClass::Private),
            ("fd12:3456::1", AddrClass::Private),
            ("::ffff:192.168.1.5", AddrClass::Private),
            ("100.64.0.1", AddrClass::Cgnat),
            ("100.127.255.1", AddrClass::Cgnat),
            ("100.128.0.1", AddrClass::Public),
            ("8.8.8.8", AddrClass::Public),
            ("2001:db8::1", AddrClass::Public),
        ] {
            assert_eq!(AddrClass::of(ip(a)), c, "{a}");
        }
        let links = vec![LinkInfo::new(ip("10.0.0.2")), LinkInfo::new(ip("8.8.8.8")), LinkInfo::new(ip("100.100.1.1")), LinkInfo::new(ip("fe80::1"))];
        let kept: Vec<_> = advertised(links).into_iter().map(|l| l.addr).collect();
        assert_eq!(kept, ["10.0.0.2", "fe80::1"]);
        // A forged class doesn't get a public address through.
        let mut forged = LinkInfo::new(ip("8.8.8.8"));
        forged.class = AddrClass::Private;
        assert!(advertised([forged]).is_empty());
    }

    #[test]
    fn measured_beats_declared_beats_nominal() {
        let mut l = LinkInfo::new(ip("10.0.0.2")).with_nominal_gbps(Some(10.0));
        assert_eq!(l.best_gbps(), Some(10.0));
        l = l.with_declared_gbps(Some(25.0));
        assert_eq!(l.best_gbps(), Some(25.0));
        l.measured.push(LinkMeasure::new("p", 18.5, 40, 1));
        assert_eq!(l.best_gbps(), Some(18.5));
    }

    #[test]
    fn local_links_are_never_public() {
        for l in local_links() {
            assert!(l.class.advertised() && l.class != AddrClass::Loopback, "{l:?}");
        }
    }
}
