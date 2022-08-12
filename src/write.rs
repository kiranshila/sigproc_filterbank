use std::marker::PhantomData;

use ux::{u1, u2, u4};

/// Creates a sigproc-compatible string
pub(crate) fn sigproc_string(s: &str) -> Vec<u8> {
    let len = s.len() as u32;
    let mut out = vec![];
    out.extend_from_slice(&len.to_ne_bytes());
    out.extend_from_slice(s.as_bytes());
    out
}

#[derive(Debug)]
pub struct WriteFilterbank<T> {
    data: Vec<u8>,
    pub telescope_id: Option<u32>,
    pub machine_id: Option<u32>,
    pub data_type: Option<u32>,
    pub raw_data_file: Option<String>,
    pub source_name: Option<String>,
    pub barycentric: Option<bool>,
    pub pulsarcentric: Option<bool>,
    pub az_start: Option<f64>,
    pub za_start: Option<f64>,
    pub src_raj: Option<f64>,
    pub src_dej: Option<f64>,
    pub tstart: Option<f64>,
    pub tsamp: Option<f64>,
    nsamples: usize,
    pub fch1: Option<f64>,
    pub foff: Option<f64>,
    nchans: usize,
    nifs: usize,
    pub ref_dm: Option<f64>,
    pub period: Option<f64>,
    pub nbeams: Option<usize>,
    pub ibeam: Option<usize>,
    // Phantom to refer to our template T
    _phantom: PhantomData<*const T>,
}

macro_rules! numeric_header_bytes {
    ($name:ident, $ctx:ident, $bytes:ident) => {
        if let Some(v) = $ctx.$name {
            $bytes.append(&mut sigproc_string(stringify!($name)));
            $bytes.extend_from_slice(&v.to_ne_bytes());
        }
    };
}

macro_rules! cast_header_bytes {
    ($name:ident, $ctx:ident, $bytes:ident, $ty:ty) => {
        if let Some(v) = $ctx.$name {
            $bytes.append(&mut sigproc_string(stringify!($name)));
            $bytes.extend_from_slice(&(v as $ty).to_ne_bytes());
        }
    };
}

macro_rules! string_header_bytes {
    ($name:ident, $ctx:ident, $bytes:ident) => {
        if let Some(v) = &$ctx.$name {
            $bytes.append(&mut sigproc_string(stringify!($name)));
            $bytes.append(&mut sigproc_string(v));
        }
    };
}

type Spectra<'a, T> = &'a [T];

pub trait PackSpectra {
    /// Turn a slice of spectra data into it's filterbank byte-representation
    fn pack(&self) -> Vec<u8>;
}

impl<'a> PackSpectra for Spectra<'a, u1> {
    fn pack(&self) -> Vec<u8> {
        self.chunks(8)
            .map(|x| {
                x.iter()
                    .enumerate()
                    // Total mystery why this has to be a u16 and not u8
                    .fold(0u8, |b, (i, v)| b | (u16::from(*v) as u8) << (8 - i - 1))
            })
            .collect()
    }
}

impl<'a> PackSpectra for Spectra<'a, u2> {
    fn pack(&self) -> Vec<u8> {
        self.chunks(4)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold(0u8, |b, (i, v)| b | u8::from(*v) << (8 - (i * 2) - 2))
            })
            .collect()
    }
}

impl<'a> PackSpectra for Spectra<'a, u4> {
    fn pack(&self) -> Vec<u8> {
        self.chunks(2)
            .map(|x| {
                x.iter()
                    .enumerate()
                    .fold(0u8, |b, (i, v)| b | u8::from(*v) << (8 - (i * 4) - 4))
            })
            .collect()
    }
}

impl<'a> PackSpectra for Spectra<'a, u8> {
    fn pack(&self) -> Vec<u8> {
        self.to_vec()
    }
}

impl<'a> PackSpectra for Spectra<'a, u16> {
    fn pack(&self) -> Vec<u8> {
        self.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }
}

impl<'a> PackSpectra for Spectra<'a, f32> {
    fn pack(&self) -> Vec<u8> {
        self.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }
}

pub trait NumBits {
    fn nbits(&self) -> usize;
}

impl NumBits for WriteFilterbank<u1> {
    fn nbits(&self) -> usize {
        1
    }
}

impl NumBits for WriteFilterbank<u2> {
    fn nbits(&self) -> usize {
        2
    }
}

impl NumBits for WriteFilterbank<u4> {
    fn nbits(&self) -> usize {
        4
    }
}

impl NumBits for WriteFilterbank<u8> {
    fn nbits(&self) -> usize {
        8
    }
}

impl NumBits for WriteFilterbank<u16> {
    fn nbits(&self) -> usize {
        16
    }
}

impl NumBits for WriteFilterbank<f32> {
    fn nbits(&self) -> usize {
        32
    }
}

impl<'a, T> WriteFilterbank<T>
where
    T: 'a,
    Spectra<'a, T>: PackSpectra,
{
    /// Creates a new, empty, Filterbank that we can write to.
    ///
    /// Data stored in this Filterbank will be written in the native endianness
    /// with data <8 bits being written such that the most significant bit is index 0.
    pub fn new(nchans: usize, nifs: usize) -> Self {
        Self {
            data: vec![],
            telescope_id: None,
            machine_id: None,
            data_type: None,
            raw_data_file: None,
            source_name: None,
            barycentric: None,
            pulsarcentric: None,
            az_start: None,
            za_start: None,
            src_raj: None,
            src_dej: None,
            tstart: None,
            tsamp: None,
            nsamples: 0,
            fch1: None,
            foff: None,
            nchans,
            nifs,
            ref_dm: None,
            period: None,
            nbeams: None,
            ibeam: None,
            _phantom: PhantomData,
        }
    }

    /// Dump the whole filterbank as bytes
    pub fn bytes(&self) -> Vec<u8>
    where
        Self: NumBits,
    {
        let mut bytes = self.header_bytes();
        bytes.extend_from_slice(&self.data);
        bytes
    }

    /// Dump the filterbank data as vector of bytes that can be streamed or written to a file
    pub fn data_bytes(&self) -> Vec<u8>
    where
        Self: NumBits,
    {
        self.data.clone()
    }

    /// Returns just the header bytes from a filterbank, can be used to start a .fil file
    pub fn header_bytes(&self) -> Vec<u8>
    where
        Self: NumBits,
    {
        let mut bytes = vec![];
        // Start
        bytes.append(&mut sigproc_string("HEADER_START"));
        // Numbers
        numeric_header_bytes!(telescope_id, self, bytes);
        numeric_header_bytes!(machine_id, self, bytes);
        numeric_header_bytes!(data_type, self, bytes);
        numeric_header_bytes!(az_start, self, bytes);
        numeric_header_bytes!(za_start, self, bytes);
        numeric_header_bytes!(src_raj, self, bytes);
        numeric_header_bytes!(src_dej, self, bytes);
        numeric_header_bytes!(tstart, self, bytes);
        numeric_header_bytes!(tsamp, self, bytes);
        numeric_header_bytes!(fch1, self, bytes);
        numeric_header_bytes!(foff, self, bytes);
        numeric_header_bytes!(ref_dm, self, bytes);
        numeric_header_bytes!(period, self, bytes);
        numeric_header_bytes!(nbeams, self, bytes);
        numeric_header_bytes!(ibeam, self, bytes);
        // Strings
        string_header_bytes!(raw_data_file, self, bytes);
        string_header_bytes!(source_name, self, bytes);
        // Things we need to cast
        cast_header_bytes!(barycentric, self, bytes, u32);
        cast_header_bytes!(pulsarcentric, self, bytes, u32);
        // Bits
        bytes.append(&mut sigproc_string("nbits"));
        bytes.extend_from_slice(&(self.nbits() as u32).to_ne_bytes());
        // Channels
        bytes.append(&mut sigproc_string("nchans"));
        bytes.extend_from_slice(&(self.nchans as u32).to_ne_bytes());
        // IFs
        bytes.append(&mut sigproc_string("nifs"));
        bytes.extend_from_slice(&(self.nifs as u32).to_ne_bytes());
        // End
        bytes.append(&mut sigproc_string("HEADER_END"));
        // Ret
        bytes
    }

    /// Push a single time-slice of spectrum data.
    ///
    /// The number of channels / nbits % 8 needs to be zero so the whole slice is byte-aligned
    pub fn push(&mut self, spectrum: Spectra<'a, T>)
    where
        Self: NumBits,
    {
        assert_eq!(
            spectrum.len(),
            self.nchans,
            "Input slice must contain every frequency"
        );
        assert_eq!(
            spectrum.len() * self.nbits() % 8,
            0,
            "Spectrum must be byte-aligned"
        );
        assert!(
            spectrum.len() * self.nbits() >= 8,
            "Total size must be at least a byte"
        );

        let mut packed = spectrum.pack();
        // Add these bytes to our data
        self.data.append(&mut packed);
        // Increment the number of samples
        self.nsamples += 1;
    }

    /// Pack a single sample of spectrum into a vector of bytes, ready to be written to a file
    pub fn pack(&self, spectrum: Spectra<'a, T>) -> Vec<u8>
    where
        Self: NumBits,
    {
        spectrum.pack()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::ReadFilterbank;

    #[test]
    fn test_sigproc_string() {
        assert_eq!(
            b"\x0C\x00\x00\x00HEADER_START".to_vec(),
            sigproc_string("HEADER_START")
        );
    }

    #[test]
    fn test_roundtrip_1_bit() {
        let mut fb = WriteFilterbank::new(8, 1);
        fb.push(&[
            u1::new(0u8),
            u1::new(1u8),
            u1::new(0u8),
            u1::new(1u8),
            u1::new(0u8),
            u1::new(1u8),
            u1::new(0u8),
            u1::new(1u8),
        ]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);
        assert_eq!(fb.get(0, 0, 2), 0f32);
        assert_eq!(fb.get(0, 0, 3), 1f32);
        assert_eq!(fb.get(0, 0, 4), 0f32);
        assert_eq!(fb.get(0, 0, 5), 1f32);
        assert_eq!(fb.get(0, 0, 6), 0f32);
        assert_eq!(fb.get(0, 0, 7), 1f32);
    }

    #[test]
    fn test_roundtrip_2_bit() {
        let mut fb = WriteFilterbank::new(4, 1);
        fb.push(&[u2::new(0u8), u2::new(1u8), u2::new(2u8), u2::new(3u8)]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        dbg!(&fb);
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);
        assert_eq!(fb.get(0, 0, 2), 2f32);
        assert_eq!(fb.get(0, 0, 3), 3f32);
    }

    #[test]
    fn test_roundtrip_4_bit() {
        let mut fb = WriteFilterbank::new(2, 1);
        fb.push(&[u4::new(1u8), u4::new(2u8)]);
        fb.push(&[u4::new(3u8), u4::new(4u8)]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 1f32);
        assert_eq!(fb.get(0, 0, 1), 2f32);
        assert_eq!(fb.get(0, 1, 0), 3f32);
        assert_eq!(fb.get(0, 1, 1), 4f32);
    }

    #[test]
    fn test_roundtrip_8_bit() {
        let mut fb = WriteFilterbank::new(2, 1);
        fb.push(&[1u8, 2u8]);
        fb.push(&[3u8, 4u8]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 1f32);
        assert_eq!(fb.get(0, 0, 1), 2f32);
        assert_eq!(fb.get(0, 1, 0), 3f32);
        assert_eq!(fb.get(0, 1, 1), 4f32);
    }

    #[test]
    fn test_roundtrip_16_bit() {
        let mut fb = WriteFilterbank::new(2, 1);
        fb.push(&[1u16, 2u16]);
        fb.push(&[3u16, 4u16]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 1f32);
        assert_eq!(fb.get(0, 0, 1), 2f32);
        assert_eq!(fb.get(0, 1, 0), 3f32);
        assert_eq!(fb.get(0, 1, 1), 4f32);
    }

    #[test]
    fn test_roundtrip_32_bit() {
        let mut fb = WriteFilterbank::new(2, 1);
        fb.push(&[1.1, 2.2]);
        fb.push(&[3.3, 4.4]);
        let bytes = fb.bytes();
        // Now parse the bytes
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 1.1f32);
        assert_eq!(fb.get(0, 0, 1), 2.2f32);
        assert_eq!(fb.get(0, 1, 0), 3.3f32);
        assert_eq!(fb.get(0, 1, 1), 4.4f32);
    }
}
