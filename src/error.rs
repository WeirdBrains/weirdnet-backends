//! Errors are `weirdnet-agent`'s [`RemoteError`], so they cross infer/2 and node/1 unchanged.

use weirdnet_agent::remote::RemoteError;

/// No such model on this node.
pub fn unknown_model(id: &str) -> RemoteError {
    RemoteError::Unavailable { message: format!("unknown model `{id}`") }
}

pub fn unavailable(message: impl Into<String>) -> RemoteError {
    RemoteError::Unavailable { message: message.into() }
}

/// The engine failed or answered with an error.
pub fn engine_error(message: impl Into<String>) -> RemoteError {
    RemoteError::Failed { message: message.into() }
}

pub fn bad_request(message: impl Into<String>) -> RemoteError {
    RemoteError::BadRequest { message: message.into() }
}
