use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;

#[derive(Debug)]
pub enum AxonError {
    Io(io::Error),
    Parse(String),
    InvalidFormat(String),
    Unsupported(String),
    State(String),
}

impl Display for AxonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AxonError::Io(err) => write!(f, "{err}"),
            AxonError::Parse(msg) => write!(f, "parse error: {msg}"),
            AxonError::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            AxonError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            AxonError::State(msg) => write!(f, "state error: {msg}"),
        }
    }
}

impl Error for AxonError {}

impl From<io::Error> for AxonError {
    fn from(value: io::Error) -> Self {
        AxonError::Io(value)
    }
}
