# sigproc_filterbank

[![license](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue?style=flat-square)](#license)
[![docs](https://img.shields.io/docsrs/sigproc_filterbank?logo=rust&style=flat-square)](https://docs.rs/sigproc_filterbank/latest/sigproc_filterbank/index.html)
[![rustc](https://img.shields.io/badge/rustc-1.57+-blue?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![build status](https://img.shields.io/github/workflow/status/kiranshila/sigproc_filterbank/CI/main?style=flat-square&logo=github)](https://github.com/kiranshila/sigproc_filterbank/actions)
[![Codecov](https://img.shields.io/codecov/c/github/kiranshila/sigproc_filterbank?style=flat-square)](https://app.codecov.io/gh/kiranshila/sigproc_filterbank)

This crate provides a simple, fast, and robust interface to the
[SIGPROC](https://sigproc.sourceforge.net/) filterbank binary format. This
format contains dynamic spectra, used commonly in pulsar and radio transient
science in radio astronomy. This implementation is pure-rust with few
dependencies and denies any unsafe code.

## Description

There are two main interfaces to filterbank files, reading and writing. These
exists as separate data structures as the way you interact with them is
fundamentally different.

### Reading

To read from a filterbank file, you simply need a slice of bytes. This can come
from a memory-mapped file, an incoming stream, etc. Then, there are getter
methods for the header data and a single `get` method for accessing points in
the dynamic spectra. Every filterbank will return f32s as the largest integer
value supported by the format (u16) fits losslessly in an f32. `get` indexes as
IF channel, Sample number, frequency channel.

```rust
use std::{fs::File, io::Read};

use sigproc_filterbank::read::ReadFilterbank;

let mut file = File::open("tests/small.fil").unwrap();
let mut bytes = vec![];
file.read_to_end(&mut bytes).unwrap();
let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
let my_spec_val = fb.get(0,0,0);
```

### Writing

To build a filterbank file, you push time-series data, just as you would receive
them in a live telescope. You must declare the number of IFs and number of
channels, and then the sample number is implied by how much data you write.
Additionally, the number of bits of the filterbank is implied by the type of
data you push. You get a compile error if you try to push different types. For
data less than 8 bits, use the [ux](https://crates.io/crates/ux) crate. We
support, u1, u2, and u4.

Then to write the file, you simply dump the bytes and write to a file, stream, etc.

```rust
use sigproc_filterbank::write::WriteFilterbank;

// Two frequency channels, one IF
let mut fb = WriteFilterbank::new(2, 1);
fb.push(&[1u8, 2u8]);
fb.push(&[3u8, 4u8]);
// Serialize to bytes
let bytes = fb.bytes();
```

### License

`sigproc_filterbank` is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) for details.
