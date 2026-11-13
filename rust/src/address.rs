//! Maps keyed by object address, and a vector push that macOS reuses.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiplier of the address hash (the 64-bit golden ratio).
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Hashes object addresses.
///
/// Addresses are aligned, so their low bits carry no information; a
/// multiplication folded onto itself spreads the high bits into the low
/// bits that pick a bucket.
#[derive(Default, Clone, Copy)]
pub(crate) struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(*byte)).wrapping_mul(GOLDEN);
        }
    }

    fn write_usize(&mut self, address: usize) {
        let mixed = (address as u64).wrapping_mul(GOLDEN);
        self.0 = mixed ^ (mixed >> 29);
    }
}

/// A map keyed by object address, hashed with [`AddressHasher`].
pub(crate) type AddressMap<V> = HashMap<usize, V, BuildHasherDefault<AddressHasher>>;

/// Pushes `item` onto `vec`, moving a full vector into a new buffer of
/// twice the capacity instead of growing it with `realloc`.
///
/// macOS keeps the large blocks that a chain of `realloc` calls frees in a
/// cache that later allocations of the same sizes do not reuse, so every
/// capture of a big graph would leave megabytes of resident memory behind.
/// A fresh allocation and a free are reused, as Python's own tables are.
pub(crate) fn push<T>(vec: &mut Vec<T>, item: T) {
    if vec.len() == vec.capacity() {
        let mut grown = Vec::with_capacity(vec.capacity().max(4) * 2);
        grown.append(vec);
        *vec = grown;
    }
    vec.push(item);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn push_keeps_every_item_in_order() {
        let mut vec = Vec::new();
        for item in 0..100 {
            push(&mut vec, item);
        }
        assert_eq!(vec, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn address_hash_spreads_aligned_addresses() {
        let build = BuildHasherDefault::<AddressHasher>::default();
        let low_bits = |address: usize| build.hash_one(address) & 0xff;
        let distinct: std::collections::HashSet<u64> =
            (0..64).map(|slot| low_bits(0x1000 + slot * 16)).collect();
        assert!(
            distinct.len() > 48,
            "only {} low-bit patterns",
            distinct.len()
        );
        let mut bytes = AddressHasher::default();
        bytes.write(b"id");
        assert_ne!(bytes.finish(), 0);
    }
}
