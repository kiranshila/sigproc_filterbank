use nom::{
    branch::alt,
    bytes::streaming::tag,
    combinator::map_res,
    multi::{length_data, length_value, many_till},
    number::{
        streaming::{f64, u32},
        Endianness,
    },
    sequence::preceded,
    IResult,
};

use crate::errors::FilterbankError;

type ParseResult<'a, T> = IResult<&'a [u8], T, FilterbankError>;
type HeaderResult<'a> = ParseResult<'a, HeaderParameter<'a>>;

fn header_string(s: &'static str, endian: Endianness) -> impl FnMut(&[u8]) -> ParseResult<&[u8]> {
    move |input: &[u8]| length_value(u32(endian), tag(s))(input)
}

fn header_start(input: &[u8], endian: Endianness) -> ParseResult<&[u8]> {
    header_string("HEADER_START", endian)(input)
}

fn header_end(input: &[u8], endian: Endianness) -> ParseResult<&[u8]> {
    header_string("HEADER_END", endian)(input)
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
/// The number of bits to represent the data
pub enum Bits {
    /// Data is 1 bit
    _1,
    /// Data is 2 bits
    _2,
    /// Data is 4 bit
    _4,
    /// Data is u8
    _8,
    /// Data is u16
    _16,
    /// Data is f32
    _32,
}

impl Bits {
    /// Number of bits each bitsize is
    pub fn bits(&self) -> usize {
        match self {
            Bits::_1 => 1,
            Bits::_2 => 2,
            Bits::_4 => 4,
            Bits::_8 => 8,
            Bits::_16 => 16,
            Bits::_32 => 32,
        }
    }

    /// Construct a [`Bits`] variant from a usize
    pub fn from_bits(bits: usize) -> Self {
        match bits {
            1 => Bits::_1,
            2 => Bits::_2,
            4 => Bits::_4,
            8 => Bits::_8,
            16 => Bits::_16,
            32 => Bits::_32,
            _ => panic!("Bad number of bits"),
        }
    }
}

/// Headers as defined by the SIGPROC specification
#[derive(Debug, PartialEq)]
pub enum HeaderParameter<'a> {
    TelescopeID(u32),
    MachineID(u32),
    DataType(u32),
    RawDataFile(&'a str),
    SourceName(&'a str),
    Barycentric(bool),
    Pulsarcentric(bool),
    AzStart(f64),
    ZaStart(f64),
    SrcRaj(f64),
    SrcDej(f64),
    TStart(f64),
    TSamp(f64),
    NBits(Bits),
    #[deprecated]
    NSamples(u32),
    FCh1(f64),
    FOff(f64),
    NChans(u32),
    NIFs(u32),
    RefDM(f64),
    Period(f64),
    NBeams(u32),
    IBeam(u32),
}

macro_rules! numeric_header {
    ($header_name:ident, $ty:ident, $param:ident) => {
        fn $header_name(input: &[u8], endian: Endianness) -> HeaderResult {
            let (remaining, value) =
                preceded(header_string(stringify!($header_name), endian), $ty(endian))(input)?;
            Ok((remaining, HeaderParameter::$param(value)))
        }
    };
}

macro_rules! boolean_header {
    ($header_name:ident, $param:ident) => {
        fn $header_name(input: &[u8], endian: Endianness) -> HeaderResult {
            let (remaining, value) =
                preceded(header_string(stringify!($header_name), endian), u32(endian))(input)?;
            Ok((remaining, HeaderParameter::$param(value == 1)))
        }
    };
}

macro_rules! string_header {
    ($header_name:ident, $param:ident) => {
        fn $header_name(input: &[u8], endian: Endianness) -> HeaderResult {
            let (remaining, value) = preceded(
                header_string(stringify!($header_name), endian),
                map_res(length_data(u32(endian)), std::str::from_utf8),
            )(input)?;
            Ok((remaining, HeaderParameter::$param(value)))
        }
    };
}

numeric_header!(telescope_id, u32, TelescopeID);
numeric_header!(machine_id, u32, MachineID);
numeric_header!(data_type, u32, DataType);
numeric_header!(nchans, u32, NChans);
numeric_header!(nifs, u32, NIFs);
numeric_header!(nbeams, u32, NBeams);
numeric_header!(ibeam, u32, IBeam);
numeric_header!(az_start, f64, AzStart);
numeric_header!(za_start, f64, ZaStart);
numeric_header!(src_raj, f64, SrcRaj);
numeric_header!(src_dej, f64, SrcDej);
numeric_header!(tstart, f64, TStart);
numeric_header!(tsamp, f64, TSamp);
numeric_header!(fch1, f64, FCh1);
numeric_header!(foff, f64, FOff);
numeric_header!(refdm, f64, RefDM);
numeric_header!(period, f64, Period);
boolean_header!(barycentric, Barycentric);
boolean_header!(pulsarcentric, Pulsarcentric);
string_header!(rawdatafile, RawDataFile);
string_header!(source_name, SourceName);

fn nbits(input: &[u8], endian: Endianness) -> HeaderResult {
    let (remaining, value) = preceded(header_string("nbits", endian), u32(endian))(input)?;
    Ok((
        remaining,
        HeaderParameter::NBits(match value {
            1 => Bits::_1,
            2 => Bits::_2,
            4 => Bits::_4,
            8 => Bits::_8,
            16 => Bits::_16,
            32 => Bits::_32,
            _ => return Err(nom::Err::Error(FilterbankError::InvalidHeader)),
        }),
    ))
}

fn header<'a>(input: &'a [u8]) -> ParseResult<'a, (Endianness, Vec<HeaderParameter<'a>>)> {
    // Determine the endianness based on the first match of HEADER_START
    let res_big = header_start(input, Endianness::Big);
    let res_little = header_start(input, Endianness::Little);
    let (remaining, endian) = if let Ok((remaining, _)) = res_big {
        (remaining, Endianness::Big)
    } else if let Ok((remaining, _)) = res_little {
        (remaining, Endianness::Little)
    } else {
        return Err(res_little.err().unwrap());
    };
    // The rest of the owl
    let (remaining, (headers, _)) = many_till(
        alt((
            // All the curried header parsers
            // For some reason, nom limits us to 21 branches per alt, so we need to nest
            alt((
                // u32s
                |i: &'a [u8]| telescope_id(i, endian),
                |i: &'a [u8]| machine_id(i, endian),
                |i: &'a [u8]| data_type(i, endian),
                |i: &'a [u8]| nbits(i, endian),
                |i: &'a [u8]| nchans(i, endian),
                |i: &'a [u8]| nifs(i, endian),
                |i: &'a [u8]| nbeams(i, endian),
                |i: &'a [u8]| ibeam(i, endian),
            )),
            alt((
                // f64s
                |i: &'a [u8]| az_start(i, endian),
                |i: &'a [u8]| za_start(i, endian),
                |i: &'a [u8]| src_raj(i, endian),
                |i: &'a [u8]| src_dej(i, endian),
                |i: &'a [u8]| tstart(i, endian),
                |i: &'a [u8]| tsamp(i, endian),
                |i: &'a [u8]| fch1(i, endian),
                |i: &'a [u8]| foff(i, endian),
                |i: &'a [u8]| refdm(i, endian),
                |i: &'a [u8]| period(i, endian),
                |i: &'a [u8]| barycentric(i, endian),
                |i: &'a [u8]| pulsarcentric(i, endian),
            )),
            alt((
                // Strings
                |i: &'a [u8]| rawdatafile(i, endian),
                |i: &'a [u8]| source_name(i, endian),
            )),
        )),
        |i: &'a [u8]| header_end(i, endian),
    )(remaining)?;
    Ok((remaining, (endian, headers)))
}

#[derive(Debug)]
/// An immutable container for a SIGPROC filterbank file
///
/// This type contains the parsed header values and can `get` data at a given sample, channel, and IF index
pub struct ReadFilterbank<'a> {
    /// Pointer to the data
    raw_data: &'a [u8],
    /// Endinanness reported from the header parser or set to native if we hold the data
    endian: Endianness,
    telescope_id: Option<u32>,
    machine_id: Option<u32>,
    data_type: Option<u32>,
    raw_data_file: Option<&'a str>,
    source_name: Option<&'a str>,
    barycentric: Option<bool>,
    pulsarcentric: Option<bool>,
    az_start: Option<f64>,
    za_start: Option<f64>,
    src_raj: Option<f64>,
    src_dej: Option<f64>,
    tstart: Option<f64>,
    tsamp: Option<f64>,
    nbits: Option<Bits>,
    nsamples: Option<usize>,
    fch1: Option<f64>,
    foff: Option<f64>,
    nchans: Option<usize>,
    nifs: Option<usize>,
    ref_dm: Option<f64>,
    period: Option<f64>,
    nbeams: Option<usize>,
    ibeam: Option<usize>,
}

impl<'a> ReadFilterbank<'a> {
    pub fn from_bytes(input: &'a [u8]) -> Result<Self, FilterbankError> {
        let (remaining, (endian, hdrs)) = header(input).map_err(|e| match e {
            nom::Err::Incomplete(_) => FilterbankError::IncompleteHeader,
            nom::Err::Error(e) => e,
            nom::Err::Failure(e) => e,
        })?;
        let mut s = Self {
            raw_data: remaining,
            endian,
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
            nbits: None,
            nsamples: None,
            fch1: None,
            foff: None,
            nchans: None,
            nifs: None,
            ref_dm: None,
            period: None,
            nbeams: None,
            ibeam: None,
        };
        // Build the fields
        for header in hdrs {
            match header {
                HeaderParameter::TelescopeID(v) => s.telescope_id = Some(v),
                HeaderParameter::MachineID(v) => s.machine_id = Some(v),
                HeaderParameter::DataType(v) => s.data_type = Some(v),
                HeaderParameter::RawDataFile(v) => s.raw_data_file = Some(v),
                HeaderParameter::SourceName(v) => s.source_name = Some(v),
                HeaderParameter::Barycentric(v) => s.barycentric = Some(v),
                HeaderParameter::Pulsarcentric(v) => s.pulsarcentric = Some(v),
                HeaderParameter::AzStart(v) => s.az_start = Some(v),
                HeaderParameter::ZaStart(v) => s.za_start = Some(v),
                HeaderParameter::SrcRaj(v) => s.src_raj = Some(v),
                HeaderParameter::SrcDej(v) => s.src_dej = Some(v),
                HeaderParameter::TStart(v) => s.tstart = Some(v),
                HeaderParameter::TSamp(v) => s.tsamp = Some(v),
                HeaderParameter::NBits(v) => s.nbits = Some(v),
                HeaderParameter::FCh1(v) => s.fch1 = Some(v),
                HeaderParameter::FOff(v) => s.foff = Some(v),
                HeaderParameter::NChans(v) => s.nchans = Some(v as usize),
                HeaderParameter::NIFs(v) => s.nifs = Some(v as usize),
                HeaderParameter::RefDM(v) => s.ref_dm = Some(v),
                HeaderParameter::Period(v) => s.period = Some(v),
                HeaderParameter::NBeams(v) => s.nbeams = Some(v as usize),
                HeaderParameter::IBeam(v) => s.ibeam = Some(v as usize),
                _ => (),
            }
        }
        // Check the invariants
        if s.nbits.is_none() || s.nchans.is_none() || s.nifs.is_none() {
            return Err(FilterbankError::IncompleteHeader);
        }
        // Compute nsamp
        let nbits = s.nbits.unwrap().bits();
        let nchans = s.nchans.unwrap();
        let nifs = s.nifs.unwrap();
        let nsamp_total = (remaining.len() * 8) / nbits;
        let nsamp = nsamp_total / nchans / nifs;
        s.nsamples = Some(nsamp);
        Ok(s)
    }

    /// Gets the value from the frequency-major data.
    ///
    /// That is, subsequent samples in frequency are contiguous in memory, arranged in blocks of time, and then in IF
    pub fn get(&self, i_if: usize, i_samp: usize, i_chan: usize) -> f32 {
        let nbits = self.nbits.unwrap().bits();
        let nifs = self.nifs.unwrap();
        let nsamp = self.nsamples.unwrap();
        let nchans = self.nchans.unwrap();
        // Bounds checks
        if i_if >= nifs {
            panic!("IF index out of bounds");
        }
        if i_samp >= nsamp {
            panic!("Sample index out of bounds");
        }
        if i_chan >= nchans {
            panic!("Channel index out of bounds");
        }
        // Stride to the right starting bit (MSB0 (hopefully, this isn't speced))
        let bit_ptr = nbits * (nchans * nsamp * i_if + nchans * i_samp + i_chan);
        // Find which byte this is in
        let byte_ptr = bit_ptr / 8usize;

        if nbits < 8 {
            // Find the bit offset in this byte
            let bit_offset = bit_ptr % 8usize;
            // Mask, cast, and return
            let shift = 8usize - nbits - bit_offset;
            ((self.raw_data[byte_ptr] >> shift) & (2u8.pow(nbits as u32) - 1u8)) as f32
        } else if nbits == 8 {
            self.raw_data[byte_ptr] as f32
        } else {
            // Grab the bytes we need
            let bytes = &self.raw_data[byte_ptr..byte_ptr + (nbits / 8usize)];
            // Convert
            match nbits {
                16 => match self.endian {
                    Endianness::Big => u16::from_be_bytes(bytes.try_into().unwrap()) as f32,
                    Endianness::Little => u16::from_le_bytes(bytes.try_into().unwrap()) as f32,
                    _ => unreachable!(),
                },
                32 => match self.endian {
                    Endianness::Big => f32::from_be_bytes(bytes.try_into().unwrap()),
                    Endianness::Little => f32::from_le_bytes(bytes.try_into().unwrap()),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            }
        }
    }

    /// 0=fake data; 1=Arecibo; 2=Ooty... others to be added
    pub fn telescope_id(&self) -> Option<u32> {
        self.telescope_id
    }

    /// 0=FAKE; 1=PSPM; 2=WAPP; 3=OOTY... others to be added
    pub fn machine_id(&self) -> Option<u32> {
        self.machine_id
    }

    /// 1=filterbank; 2=time series... others to be added
    pub fn data_type(&self) -> Option<u32> {
        self.data_type
    }

    /// The name of the original data file
    pub fn raw_data_file(&self) -> Option<&str> {
        self.raw_data_file
    }

    /// The name of the source being observed by the telescope
    pub fn source_name(&self) -> Option<&str> {
        self.source_name
    }

    /// True is the data is Barycentric
    pub fn barycentric(&self) -> Option<bool> {
        self.barycentric
    }

    /// True is the data is Pulsar-centric
    pub fn pulsarcentric(&self) -> Option<bool> {
        self.pulsarcentric
    }

    /// Telescope azimuth at start of scan (degrees)
    pub fn az_start(&self) -> Option<f64> {
        self.az_start
    }

    /// Telescope zenith angle at start of scan (degrees)
    pub fn za_start(&self) -> Option<f64> {
        self.za_start
    }

    /// Right ascension (J2000) of source (hhmmss.s)
    pub fn src_raj(&self) -> Option<f64> {
        self.src_raj
    }

    /// Declination (J2000) of source (ddmmss.s)
    pub fn src_dej(&self) -> Option<f64> {
        self.src_dej
    }

    /// Timestamp (MJD) of first sample
    pub fn tstart(&self) -> Option<f64> {
        self.tstart
    }

    /// Time interval between samples (s)
    pub fn tsamp(&self) -> Option<f64> {
        self.tsamp
    }

    /// Center frequency of first channel (MHz)
    pub fn fch1(&self) -> Option<f64> {
        self.fch1
    }

    /// Channel bandwidth (MHz)
    pub fn foff(&self) -> Option<f64> {
        self.foff
    }

    /// Number of bits per time sample
    pub fn nbits(&self) -> Bits {
        self.nbits.unwrap()
    }

    /// Number of time samples in the data file
    pub fn nsamples(&self) -> usize {
        self.nsamples.unwrap()
    }

    /// Number of filterbank channels
    pub fn nchans(&self) -> usize {
        self.nchans.unwrap()
    }

    /// Number of separate IF channels
    pub fn nifs(&self) -> usize {
        self.nifs.unwrap()
    }

    /// Reference dispersion measure (cm^-3 pc)
    pub fn ref_dm(&self) -> Option<f64> {
        self.ref_dm
    }

    /// Folding period (s)
    pub fn period(&self) -> Option<f64> {
        self.period
    }

    /// Numbers of beams
    pub fn nbeams(&self) -> Option<usize> {
        self.nbeams
    }

    /// Current beam number
    pub fn ibeam(&self) -> Option<usize> {
        self.ibeam
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use memmap2::Mmap;

    use super::*;
    use crate::write::sigproc_string;

    #[test]
    fn test_start_end() {
        let hstart = sigproc_string("HEADER_START");
        let hend = sigproc_string("HEADER_END");
        let (_, s) = header_start(&hstart, Endianness::Native).unwrap();
        let (_, e) = header_end(&hend, Endianness::Native).unwrap();
        assert_eq!(b"HEADER_START", s);
        assert_eq!(b"HEADER_END", e);
    }

    #[test]
    fn test_numeric_headers() {
        let mut az = sigproc_string("az_start");
        az.extend_from_slice(&123.456f64.to_ne_bytes());
        let (_, res) = az_start(&az, Endianness::Native).unwrap();
        assert_eq!(HeaderParameter::AzStart(123.456), res);
    }

    #[test]
    fn test_boolean_headers() {
        // True
        let mut bc = sigproc_string("barycentric");
        bc.extend_from_slice(&1u32.to_ne_bytes());
        let (_, res) = barycentric(&bc, Endianness::Native).unwrap();
        assert_eq!(HeaderParameter::Barycentric(true), res);
        // False
        let mut bc = sigproc_string("barycentric");
        bc.extend_from_slice(&0u32.to_ne_bytes());
        let (_, res) = barycentric(&bc, Endianness::Native).unwrap();
        assert_eq!(HeaderParameter::Barycentric(false), res);
    }

    #[test]
    fn test_string_headers() {
        let mut pair = sigproc_string("source_name");
        let name = sigproc_string("SGR A*");
        pair.extend_from_slice(&name);
        let (_, source) = source_name(&pair, Endianness::Native).unwrap();
        assert_eq!(HeaderParameter::SourceName("SGR A*"), source);
    }

    #[test]
    fn test_header() {
        let mut hdr = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&8u32.to_ne_bytes());
        let mut azstart = sigproc_string("az_start");
        azstart.extend_from_slice(&867.5309f64.to_ne_bytes());
        let mut zastart = sigproc_string("za_start");
        zastart.extend_from_slice(&420.69f64.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        hdr.extend_from_slice(&nb);
        hdr.extend_from_slice(&azstart);
        hdr.extend_from_slice(&zastart);
        hdr.extend_from_slice(&end);
        let (_, (_, res)) = header(&hdr).unwrap();
        assert_eq!(
            vec![
                HeaderParameter::NBits(Bits::_8),
                HeaderParameter::AzStart(867.5309),
                HeaderParameter::ZaStart(420.69),
            ],
            res
        );
    }

    #[test]
    fn test_header_from_fil() {
        let file = File::open("tests/header.fil").unwrap();
        let bytes = unsafe { Mmap::map(&file).unwrap() };
        let (_, (_, res)) = header(&bytes[..]).unwrap();
        assert_eq!(
            vec![
                HeaderParameter::TelescopeID(4),
                HeaderParameter::NBits(Bits::_32),
                HeaderParameter::SourceName("J2048-1616"),
                HeaderParameter::DataType(1),
                HeaderParameter::NChans(33554432),
                HeaderParameter::IBeam(1),
                HeaderParameter::Barycentric(false),
                HeaderParameter::Pulsarcentric(false),
                HeaderParameter::TSamp(16.777216),
                HeaderParameter::FOff(3.814697265625e-6),
                HeaderParameter::SrcRaj(204835.6),
                HeaderParameter::SrcDej(-158355.5),
                HeaderParameter::TStart(58625.737858796296,),
                HeaderParameter::NBeams(1),
                HeaderParameter::FCh1(704.0),
                HeaderParameter::NIFs(1),
            ],
            res
        );
    }

    #[test]
    fn test_1_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&1u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        let data: [u8; 1] = [0b0001_1011];
        bytes.extend_from_slice(&data);
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 0f32);

        assert_eq!(fb.get(0, 1, 0), 0f32);
        assert_eq!(fb.get(0, 1, 1), 1f32);

        assert_eq!(fb.get(1, 0, 0), 1f32);
        assert_eq!(fb.get(1, 0, 1), 0f32);

        assert_eq!(fb.get(1, 1, 0), 1f32);
        assert_eq!(fb.get(1, 1, 1), 1f32);
    }

    #[test]
    fn test_2_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&2u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        let data: [u8; 2] = [0b00_01_10_11, 0b11_10_01_00];
        bytes.extend_from_slice(&data);
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);

        assert_eq!(fb.get(0, 1, 0), 2f32);
        assert_eq!(fb.get(0, 1, 1), 3f32);

        assert_eq!(fb.get(1, 0, 0), 3f32);
        assert_eq!(fb.get(1, 0, 1), 2f32);

        assert_eq!(fb.get(1, 1, 0), 1f32);
        assert_eq!(fb.get(1, 1, 1), 0f32);
    }

    #[test]
    fn test_4_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&4u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        let data: [u8; 4] = [0b0000_0001, 0b0010_0011, 0b0100_0101, 0b0110_0111];
        bytes.extend_from_slice(&data);
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);

        assert_eq!(fb.get(0, 1, 0), 2f32);
        assert_eq!(fb.get(0, 1, 1), 3f32);

        assert_eq!(fb.get(1, 0, 0), 4f32);
        assert_eq!(fb.get(1, 0, 1), 5f32);

        assert_eq!(fb.get(1, 1, 0), 6f32);
        assert_eq!(fb.get(1, 1, 1), 7f32);
    }

    #[test]
    fn test_8_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&8u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        let data: [u8; 8] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
        bytes.extend_from_slice(&data);
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);

        assert_eq!(fb.get(0, 1, 0), 2f32);
        assert_eq!(fb.get(0, 1, 1), 3f32);

        assert_eq!(fb.get(1, 0, 0), 4f32);
        assert_eq!(fb.get(1, 0, 1), 5f32);

        assert_eq!(fb.get(1, 1, 0), 6f32);
        assert_eq!(fb.get(1, 1, 1), 7f32);
    }

    #[test]
    fn test_16_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&16u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        bytes.extend_from_slice(&0u16.to_ne_bytes());
        bytes.extend_from_slice(&1u16.to_ne_bytes());
        bytes.extend_from_slice(&2u16.to_ne_bytes());
        bytes.extend_from_slice(&3u16.to_ne_bytes());
        bytes.extend_from_slice(&4u16.to_ne_bytes());
        bytes.extend_from_slice(&5u16.to_ne_bytes());
        bytes.extend_from_slice(&6u16.to_ne_bytes());
        bytes.extend_from_slice(&7u16.to_ne_bytes());
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0f32);
        assert_eq!(fb.get(0, 0, 1), 1f32);

        assert_eq!(fb.get(0, 1, 0), 2f32);
        assert_eq!(fb.get(0, 1, 1), 3f32);

        assert_eq!(fb.get(1, 0, 0), 4f32);
        assert_eq!(fb.get(1, 0, 1), 5f32);

        assert_eq!(fb.get(1, 1, 0), 6f32);
        assert_eq!(fb.get(1, 1, 1), 7f32);
    }

    #[test]
    fn test_32_bit_data() {
        // Build header
        let mut bytes = sigproc_string("HEADER_START");
        let mut nb = sigproc_string("nbits");
        nb.extend_from_slice(&32u32.to_ne_bytes());
        let mut nc = sigproc_string("nchans");
        nc.extend_from_slice(&2u32.to_ne_bytes());
        let mut ni = sigproc_string("nifs");
        ni.extend_from_slice(&2u32.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        bytes.extend_from_slice(&nb);
        bytes.extend_from_slice(&nc);
        bytes.extend_from_slice(&ni);
        bytes.extend_from_slice(&end);
        // Build data (2 samples total over 2 channels over 2 if)
        bytes.extend_from_slice(&0.0f32.to_ne_bytes());
        bytes.extend_from_slice(&0.1f32.to_ne_bytes());
        bytes.extend_from_slice(&0.2f32.to_ne_bytes());
        bytes.extend_from_slice(&0.3f32.to_ne_bytes());
        bytes.extend_from_slice(&0.4f32.to_ne_bytes());
        bytes.extend_from_slice(&0.5f32.to_ne_bytes());
        bytes.extend_from_slice(&0.6f32.to_ne_bytes());
        bytes.extend_from_slice(&0.7f32.to_ne_bytes());
        // Parse and build
        let fb = ReadFilterbank::from_bytes(&bytes).unwrap();
        assert_eq!(fb.get(0, 0, 0), 0.0f32);
        assert_eq!(fb.get(0, 0, 1), 0.1f32);

        assert_eq!(fb.get(0, 1, 0), 0.2f32);
        assert_eq!(fb.get(0, 1, 1), 0.3f32);

        assert_eq!(fb.get(1, 0, 0), 0.4f32);
        assert_eq!(fb.get(1, 0, 1), 0.5f32);

        assert_eq!(fb.get(1, 1, 0), 0.6f32);
        assert_eq!(fb.get(1, 1, 1), 0.7f32);
    }

    #[test]
    fn test_freq_avg_from_fb() {
        // nsamp - 10, nbits - 8 , nifs - 1, nchans = 336,
        let file = File::open("tests/small.fil").unwrap();
        let bytes = unsafe { Mmap::map(&file).unwrap() };
        let fb = ReadFilterbank::from_bytes(&bytes[..]).unwrap();
        let mut tm = vec![0f32; fb.nsamples()];
        (0..tm.len()).for_each(|j| {
            for i in 0..fb.nchans() {
                tm[j] += fb.get(0, j, i) / 10f32;
            }
        });
        assert_eq!(tm, [
            4310.1016, 4280.099, 4295.5977, 4328.5996, 4326.4004, 4274.6997, 4338.1997, 4332.3013,
            4384.0015, 4316.0996
        ]);
    }
}
