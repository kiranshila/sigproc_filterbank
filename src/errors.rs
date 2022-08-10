use nom::error::{FromExternalError, ParseError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FilterbankError {
    #[error("Invalid header")]
    InvalidHeader,
    #[error("Incomplete header")]
    IncompleteHeader,
    #[error("Extenal parser error")]
    ExternalParseError(String),
    #[error("Unknown parser error")]
    Unparseable,
}

impl<I> ParseError<I> for FilterbankError {
    fn from_error_kind(_input: I, _kind: nom::error::ErrorKind) -> Self {
        FilterbankError::Unparseable
    }

    fn append(_: I, _: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
}

impl<I, E> FromExternalError<I, E> for FilterbankError
where
    E: std::error::Error,
{
    fn from_external_error(_input: I, _kind: nom::error::ErrorKind, e: E) -> Self {
        FilterbankError::ExternalParseError(format!("{}", e))
    }
}
