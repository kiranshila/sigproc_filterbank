use std::collections::HashSet;

use derivative::Derivative;
use nom::{
    branch::alt,
    bytes::streaming::tag,
    combinator::map_res,
    multi::{length_data, length_value, many_till},
    number::{
        streaming::{f64, u32},
        Endianness,
    },
    sequence::{preceded, terminated},
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

/// Headers as defined by the SIGPROC specification
#[derive(Derivative)]
#[derivative(Debug, PartialEq, Eq, Hash)]
pub enum HeaderParameter<'a> {
    /// 0=fake data; 1=Arecibo; 2=Ooty... others to be added
    TelescopeID(u32),
    /// 0=FAKE; 1=PSPM; 2=WAPP; 3=OOTY... others to be added
    MachineID(u32),
    /// 1=filterbank; 2=time series... others to be added
    DataType(u32),
    /// The name of the original data file
    RawDataFile(&'a str),
    /// The name of the source being observed by the telescope
    SourceName(&'a str),
    /// True is the data is Barycentric
    Barycentric(bool),
    /// True is the data is Pulsar-centric
    Pulsarcentric(bool),
    /// Telescope azimuth at start of scan (degrees)
    AzStart(#[derivative(Hash = "ignore")] f64),
    /// Telescope zenith angle at start of scan (degrees)
    ZaStart(#[derivative(Hash = "ignore")] f64),
    /// Right ascension (J2000) of source (hhmmss.s)
    SrcRaj(#[derivative(Hash = "ignore")] f64),
    /// Declination (J2000) of source (ddmmss.s)
    SrcDej(#[derivative(Hash = "ignore")] f64),
    /// Timestamp (MJD) of first sample
    TStart(#[derivative(Hash = "ignore")] f64),
    /// Time interval between samples (s)
    TSamp(#[derivative(Hash = "ignore")] f64),
    /// Number of bits per time sample
    NBits(u32),
    /// Number of time samples in the data file
    #[deprecated]
    NSamples(u32),
    /// Center frequency of first channel (MHz)
    FCh1(#[derivative(Hash = "ignore")] f64),
    /// Channel bandwidth (MHz)
    FOff(#[derivative(Hash = "ignore")] f64),
    /// Number of filterbank channels
    NChans(u32),
    /// Number of separate IF channels
    NIFs(u32),
    /// Reference dispersion measure (cm^-3 pc)
    RefDM(#[derivative(Hash = "ignore")] f64),
    /// Folding period (s)
    Period(#[derivative(Hash = "ignore")] f64),
    /// Numbers of beams
    NBeams(u32),
    /// Current beam number
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
numeric_header!(nbits, u32, NBits);
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

pub fn header<'a>(input: &'a [u8]) -> ParseResult<'a, HashSet<HeaderParameter<'a>>> {
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
                |i: &'a [u8]| rawdatafile(i, endian),
                |i: &'a [u8]| source_name(i, endian),
            )),
        )),
        |i: &'a [u8]| header_end(i, endian),
    )(remaining)?;
    Ok((remaining, headers.into_iter().collect()))
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use memmap2::Mmap;

    use super::*;
    use crate::serialization::sigproc_string;

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
        nb.extend_from_slice(&4u32.to_ne_bytes());
        let mut azstart = sigproc_string("az_start");
        azstart.extend_from_slice(&867.5309f64.to_ne_bytes());
        let mut zastart = sigproc_string("za_start");
        zastart.extend_from_slice(&420.69f64.to_ne_bytes());
        let end = sigproc_string("HEADER_END");
        hdr.extend_from_slice(&nb);
        hdr.extend_from_slice(&azstart);
        hdr.extend_from_slice(&zastart);
        hdr.extend_from_slice(&end);
        let (_, res) = header(&hdr).unwrap();
        assert_eq!(
            HashSet::from_iter(
                vec![
                    HeaderParameter::AzStart(867.5309),
                    HeaderParameter::ZaStart(420.69),
                    HeaderParameter::NBits(4)
                ]
                .into_iter()
            ),
            res
        );
    }

    #[test]
    fn test_header_from_fil() {
        let file = File::open("tests/header.fil").unwrap();
        let bytes = unsafe { Mmap::map(&file).unwrap() };
        let (_, res) = header(&bytes[..]).unwrap();
        assert_eq!(
            HashSet::from_iter(
                vec![
                    HeaderParameter::DataType(1),
                    HeaderParameter::IBeam(1),
                    HeaderParameter::FOff(3.814697265625e-6),
                    HeaderParameter::Pulsarcentric(false),
                    HeaderParameter::TelescopeID(4),
                    HeaderParameter::TStart(58625.737858796296,),
                    HeaderParameter::NBits(32),
                    HeaderParameter::SourceName("J2048-1616"),
                    HeaderParameter::Barycentric(false),
                    HeaderParameter::NBeams(1),
                    HeaderParameter::NChans(33554432),
                    HeaderParameter::TSamp(16.777216),
                    HeaderParameter::FCh1(704.0),
                    HeaderParameter::SrcDej(-158355.5),
                    HeaderParameter::SrcRaj(204835.6),
                    HeaderParameter::NIFs(1),
                ]
                .into_iter()
            ),
            res
        );
    }
}
