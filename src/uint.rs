//! Root module for 256-bit unsigned integer type.

// use alloc::vec::Vec;
use core::num::ParseIntError;

use rayon::{
    iter::plumbing::{bridge, Producer},
    prelude::*,
};

extern crate std;
use std::vec::Vec;

use crate::I256;

pub use self::convert::AsU256;

mod api;
mod cmp;
mod convert;
mod fmt;
mod iter;
mod ops;
mod parse;

/// A 256-bit unsigned integer type.
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct U256(pub [u128; 2]);

impl U256 {
    /// The additive identity for this integer type, i.e. `0`.
    pub const ZERO: Self = U256([0; 2]);

    /// The multiplicative identity for this integer type, i.e. `1`.
    pub const ONE: Self = U256::new(1);

    /// Creates a new 256-bit integer value from a primitive `u128` integer.
    #[inline]
    pub const fn new(value: u128) -> Self {
        U256::from_words(0, value)
    }

    /// Creates a new 256-bit integer value from high and low words.
    #[inline]
    pub const fn from_words(hi: u128, lo: u128) -> Self {
        #[cfg(target_endian = "little")]
        {
            U256([lo, hi])
        }
        #[cfg(target_endian = "big")]
        {
            U256([hi, lo])
        }
    }

    /// Splits a 256-bit integer into high and low words.
    #[inline]
    pub const fn into_words(self) -> (u128, u128) {
        #[cfg(target_endian = "little")]
        {
            let U256([lo, hi]) = self;
            (hi, lo)
        }
        #[cfg(target_endian = "big")]
        {
            let U256([hi, lo]) = self;
            (hi, lo)
        }
    }

    /// Get the low 128-bit word for this unsigned integer.
    #[inline]
    pub fn low(&self) -> &u128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[1]
        }
    }

    /// Get the low 128-bit word for this unsigned integer as a mutable
    /// reference.
    #[inline]
    pub fn low_mut(&mut self) -> &mut u128 {
        #[cfg(target_endian = "little")]
        {
            &mut self.0[0]
        }
        #[cfg(target_endian = "big")]
        {
            &mut self.0[1]
        }
    }

    /// Get the high 128-bit word for this unsigned integer.
    #[inline]
    pub fn high(&self) -> &u128 {
        #[cfg(target_endian = "little")]
        {
            &self.0[1]
        }
        #[cfg(target_endian = "big")]
        {
            &self.0[0]
        }
    }

    /// Get the high 128-bit word for this unsigned integer as a mutable
    /// reference.
    #[inline]
    pub fn high_mut(&mut self) -> &mut u128 {
        #[cfg(target_endian = "little")]
        {
            &mut self.0[1]
        }
        #[cfg(target_endian = "big")]
        {
            &mut self.0[0]
        }
    }

    /// Converts a prefixed string slice in base 16 to an integer.
    ///
    /// The string is expected to be an optional `+` sign followed by the `0x`
    /// prefix and finally the digits. Leading and trailing whitespace represent
    /// an error.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::from_str_hex("0x2A"), Ok(U256::new(42)));
    /// ```
    pub fn from_str_hex(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_radix(src, 16, Some("0x"))
    }

    /// Converts a prefixed string slice in a base determined by the prefix to
    /// an integer.
    ///
    /// The string is expected to be an optional `+` sign followed by the one of
    /// the supported prefixes and finally the digits. Leading and trailing
    /// whitespace represent an error. The base is dertermined based on the
    /// prefix:
    ///
    /// * `0x`: base `16`
    /// * no prefix: base `10`
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// # use ethnum::U256;
    /// assert_eq!(U256::from_str_prefixed("42"), Ok(U256::new(42)));
    /// assert_eq!(U256::from_str_prefixed("0xa"), Ok(U256::new(10)));
    /// ```
    pub fn from_str_prefixed(src: &str) -> Result<Self, ParseIntError> {
        crate::parse::from_str_prefixed(src)
    }

    /// Cast to a primitive `i8`.
    pub const fn as_i8(self) -> i8 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i16`.
    pub const fn as_i16(self) -> i16 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i32`.
    pub const fn as_i32(self) -> i32 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i64`.
    pub const fn as_i64(self) -> i64 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `i128`.
    pub const fn as_i128(self) -> i128 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a `I256`.
    pub const fn as_i256(self) -> I256 {
        let Self([a, b]) = self;
        I256([a as _, b as _])
    }

    /// Cast to a primitive `u8`.
    pub const fn as_u8(self) -> u8 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u16`.
    pub const fn as_u16(self) -> u16 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u32`.
    pub const fn as_u32(self) -> u32 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u64`.
    pub const fn as_u64(self) -> u64 {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `u128`.
    pub const fn as_u128(self) -> u128 {
        let (_, lo) = self.into_words();
        lo
    }

    /// Cast to a primitive `isize`.
    pub const fn as_isize(self) -> isize {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `usize`.
    pub const fn as_usize(self) -> usize {
        let (_, lo) = self.into_words();
        lo as _
    }

    /// Cast to a primitive `f32`.
    pub fn as_f32(self) -> f32 {
        match self.into_words() {
            (0, lo) => lo as _,
            _ => f32::INFINITY,
        }
    }

    /// Cast to a primitive `f64`.
    pub fn as_f64(self) -> f64 {
        // NOTE: Binary representation of 2**128. This is used because `powi` is
        // neither `const` nor `no_std`.
        const HI: u64 = 0x47f0000000000000;
        let (hi, lo) = self.into_words();
        (hi as f64) * f64::from_bits(HI) + (lo as f64)
    }
}

/// range for U256
pub struct U256Range {
    start: U256,
    end: U256,
}

impl U256Range {
    fn new(start: U256, end: U256) -> Self {
        Self { start, end }
    }
}

impl Iterator for U256Range {
    type Item = U256;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let result = self.start;
            self.start += U256::new(1);
            Some(result)
        } else {
            None
        }
    }
}

type Data = U256;

/// parallel iter for U256
pub struct DataCollection {
    /// container for U256
    pub data: Vec<Data>,
}

impl<'a> IntoParallelIterator for &'a DataCollection {
    type Iter = ParDataIter<'a>;
    type Item = &'a Data;

    fn into_par_iter(self) -> Self::Iter {
        ParDataIter { data: &self.data }
    }
}

impl<'a> IntoParallelIterator for &'a mut DataCollection {
    type Iter = ParDataIterMut<'a>;
    type Item = &'a mut Data;

    fn into_par_iter(self) -> Self::Iter {
        ParDataIterMut { data: self }
    }
}

/// U256 parallel iter collector
impl DataCollection {
    /// constructor
    pub fn new<I>(data: I) -> Self
    where
        I: IntoIterator<Item = Data>,
    {
        Self {
            data: data.into_iter().collect(),
        }
    }
}

pub struct ParDataIter<'a> {
    data: &'a [Data],
}

pub struct ParDataIterMut<'a> {
    data: &'a mut DataCollection,
}

impl<'a> ParallelIterator for ParDataIter<'a> {
    type Item = &'a Data;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(<Self as IndexedParallelIterator>::len(self))
    }
}

impl<'a> ParallelIterator for ParDataIterMut<'a> {
    type Item = &'a mut Data;
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(<Self as IndexedParallelIterator>::len(self))
    }
}

impl<'a> IndexedParallelIterator for ParDataIter<'a> {
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        let data_producer = DataProducer::from(self);
        callback.callback(data_producer)
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> IndexedParallelIterator for ParDataIterMut<'a> {
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        let producer = DataProducerMut::from(self);
        callback.callback(producer)
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.data.data.len()
    }
}

pub struct DataProducer<'a> {
    data_slice: &'a [Data],
}

pub struct DataProducerMut<'a> {
    data_slice: &'a mut [Data],
}

impl<'a> From<&'a mut [Data]> for DataProducerMut<'a> {
    fn from(data_slice: &'a mut [Data]) -> Self {
        Self { data_slice }
    }
}

impl<'a> From<ParDataIter<'a>> for DataProducer<'a> {
    fn from(iterator: ParDataIter<'a>) -> Self {
        Self {
            data_slice: &iterator.data,
        }
    }
}

impl<'a> From<ParDataIterMut<'a>> for DataProducerMut<'a> {
    fn from(iterator: ParDataIterMut<'a>) -> Self {
        Self {
            data_slice: &mut iterator.data.data,
        }
    }
}

impl<'a> Producer for DataProducer<'a> {
    type Item = &'a Data;
    type IntoIter = std::slice::Iter<'a, Data>;

    fn into_iter(self) -> Self::IntoIter {
        self.data_slice.iter()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.data_slice.split_at(index);
        (
            DataProducer { data_slice: left },
            DataProducer { data_slice: right },
        )
    }
}

impl<'a> Producer for DataProducerMut<'a> {
    type Item = &'a mut Data;
    type IntoIter = std::slice::IterMut<'a, Data>;

    fn into_iter(self) -> Self::IntoIter {
        self.data_slice.iter_mut()
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.data_slice.split_at_mut(index);
        (Self::from(left), Self::from(right))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rayon::prelude::*;

    use crate::uint::{DataCollection, U256Range, U256};

    #[test]
    #[allow(clippy::float_cmp)]
    fn converts_to_f64() {
        assert_eq!(U256::from_words(1, 0).as_f64(), 2.0f64.powi(128))
    }

    #[test]
    fn iterates() {
        let range = U256Range::new(
            U256::from_str_hex("0x0").unwrap(),
            U256::from_str_hex("0x2").unwrap(),
        );
        let mut iter = range.into_iter();
        assert_eq!(iter.next(), Some(U256::new(0)));
        assert_eq!(iter.next(), Some(U256::new(1)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn parallel_iterates() {
        let data = DataCollection::new(vec![
            U256::new(0),
            U256::new(1),
            U256::new(2),
        ]);
        let res: Vec<i32> = data.into_par_iter()
            .map(|x| match x.as_i32() {
                0 => 0,
                1 => 1,
                _ => -1,
            }).collect();
        assert_eq!(res, vec![0, 1, -1]);
    }
}
