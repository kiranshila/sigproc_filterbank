/// Creates a sigproc-compatible string
pub(crate) fn sigproc_string(s: &str) -> Vec<u8> {
    let len = s.len() as u32;
    let mut out = vec![];
    out.extend_from_slice(&len.to_ne_bytes());
    out.extend_from_slice(s.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigproc_string() {
        assert_eq!(
            b"\x0C\x00\x00\x00HEADER_START".to_vec(),
            sigproc_string("HEADER_START")
        );
    }
}
