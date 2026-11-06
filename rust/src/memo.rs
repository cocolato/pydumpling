//! An insertion-ordered identity memo with transactional rollback.
//!
//! The pure-Python walk memoizes every captured value in a dict keyed by
//! `id(original)`. [`Memo`] keeps the same observable order: a new key is
//! appended and replacing a key keeps its position. Between [`Memo::begin`]
//! and [`Memo::commit`] every change is journaled, so [`Memo::rollback`]
//! restores the memo exactly, order included, when the walker hands a call
//! back to Python.

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
pub struct AddressHasher(u64);

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

/// Pushes `item` onto `vec`, moving a full vector into a new buffer of
/// twice the capacity instead of growing it with `realloc`.
///
/// macOS keeps the large blocks that a chain of `realloc` calls frees in a
/// cache that later allocations of the same sizes do not reuse, so every
/// capture of a big graph would leave megabytes of resident memory behind.
/// A fresh allocation and a free are reused, as Python's own tables are.
///
/// ```
/// let mut vec = Vec::new();
/// for item in 0..100 {
///     _native::memo::push(&mut vec, item);
/// }
/// assert_eq!(vec, (0..100).collect::<Vec<_>>());
/// ```
pub fn push<T>(vec: &mut Vec<T>, item: T) {
    if vec.len() == vec.capacity() {
        let mut grown = Vec::with_capacity(vec.capacity().max(4) * 2);
        grown.append(vec);
        *vec = grown;
    }
    vec.push(item);
}

/// A map keyed by object address, hashed with [`AddressHasher`].
pub type AddressMap<V> = HashMap<usize, V, BuildHasherDefault<AddressHasher>>;

/// A map from object address to entry index.
type Index = AddressMap<usize>;

/// An insertion-ordered map from object address to `T`, with rollback.
///
/// ```
/// use _native::memo::Memo;
///
/// let mut memo = Memo::new();
/// memo.insert(16, "a");
/// memo.begin();
/// memo.insert(32, "b");
/// memo.insert(16, "c");
/// assert_eq!(memo.iter().collect::<Vec<_>>(), [(16, &"c"), (32, &"b")]);
/// memo.rollback();
/// assert_eq!(memo.iter().collect::<Vec<_>>(), [(16, &"a")]);
/// ```
pub struct Memo<T> {
    index: Index,
    entries: Vec<(usize, T)>,
    /// The entry count when the current transaction began.
    mark: Option<usize>,
    /// Entries older than `mark` replaced in this transaction, oldest first.
    undo: Vec<(usize, T)>,
}

impl<T> Default for Memo<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Memo<T> {
    /// An empty memo outside any transaction.
    pub fn new() -> Self {
        Self {
            index: Index::default(),
            entries: Vec::new(),
            mark: None,
            undo: Vec::new(),
        }
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memo holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry for `key`.
    pub fn get(&self, key: usize) -> Option<&T> {
        self.index.get(&key).map(|&slot| &self.entries[slot].1)
    }

    /// Whether `key` has an entry.
    pub fn contains(&self, key: usize) -> bool {
        self.index.contains_key(&key)
    }

    /// Sets the entry for `key`: appended if new, replaced in place if not.
    /// Returns the position of a replaced entry.
    pub fn insert(&mut self, key: usize, value: T) -> Option<usize> {
        match self.index.get(&key) {
            Some(&slot) => {
                let old = std::mem::replace(&mut self.entries[slot].1, value);
                if self.mark.is_some_and(|mark| slot < mark) {
                    self.undo.push((slot, old));
                }
                Some(slot)
            }
            None => {
                self.index.insert(key, self.entries.len());
                push(&mut self.entries, (key, value));
                None
            }
        }
    }

    /// Starts journaling changes, ending any transaction in progress.
    pub fn begin(&mut self) {
        self.mark = Some(self.entries.len());
        self.undo.clear();
    }

    /// Keeps every change since [`Memo::begin`].
    pub fn commit(&mut self) {
        self.mark = None;
        self.undo.clear();
    }

    /// Undoes every change since [`Memo::begin`]; no-op outside one.
    pub fn rollback(&mut self) {
        let Some(mark) = self.mark.take() else {
            return;
        };
        while let Some((slot, old)) = self.undo.pop() {
            self.entries[slot].1 = old;
        }
        for (key, _) in self.entries.drain(mark..) {
            self.index.remove(&key);
        }
    }

    /// Entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.iter_from(0)
    }

    /// Entries in insertion order from position `start` on.
    ///
    /// ```
    /// let mut memo = _native::memo::Memo::new();
    /// memo.insert(16, "a");
    /// memo.insert(32, "b");
    /// assert_eq!(memo.iter_from(1).collect::<Vec<_>>(), [(32, &"b")]);
    /// assert_eq!(memo.iter_from(5).count(), 0);
    /// ```
    pub fn iter_from(&self, start: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = start.min(self.entries.len());
        self.entries[start..]
            .iter()
            .map(|(key, value)| (*key, value))
    }

    /// Removes every entry and ends any transaction.
    pub fn clear(&mut self) {
        self.index.clear();
        self.entries.clear();
        self.mark = None;
        self.undo.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn replacing_a_key_keeps_its_position() {
        let mut memo = Memo::new();
        assert_eq!(memo.insert(1, 'a'), None);
        assert_eq!(memo.insert(2, 'b'), None);
        assert_eq!(memo.insert(1, 'c'), Some(0));
        assert_eq!(memo.iter().collect::<Vec<_>>(), [(1, &'c'), (2, &'b')]);
        assert_eq!(memo.len(), 2);
        assert!(!memo.is_empty());
        assert!(memo.contains(2));
        assert_eq!(memo.get(3), None);
    }

    #[test]
    fn rollback_restores_replaced_entries_and_drops_new_ones() {
        let mut memo = Memo::default();
        memo.insert(1, 10);
        memo.insert(2, 20);
        memo.begin();
        memo.insert(1, 11);
        memo.insert(1, 12);
        memo.insert(3, 30);
        memo.insert(3, 31);
        memo.rollback();
        assert_eq!(memo.iter().collect::<Vec<_>>(), [(1, &10), (2, &20)]);
        assert!(!memo.contains(3));
        // Outside a transaction, rollback changes nothing.
        memo.insert(4, 40);
        memo.rollback();
        assert!(memo.contains(4));
    }

    #[test]
    fn commit_keeps_changes_and_clear_empties() {
        let mut memo = Memo::new();
        memo.begin();
        memo.insert(5, 'x');
        memo.commit();
        memo.rollback();
        assert_eq!(memo.get(5), Some(&'x'));
        memo.clear();
        assert!(memo.is_empty());
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
