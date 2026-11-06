//! Property test: `Memo` behaves like a Python dict keyed by `id()`, with a
//! transaction that rollback restores exactly, order included.

use _native::memo::Memo;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Op {
    Insert(usize, u32),
    Begin,
    Commit,
    Rollback,
    Clear,
}

fn op() -> impl Strategy<Value = Op> {
    // Few distinct keys, so replacements are common; aligned like addresses.
    let key = (0usize..12).prop_map(|slot| 0x7f00_0000 + slot * 16);
    prop_oneof![
        8 => (key, any::<u32>()).prop_map(|(key, value)| Op::Insert(key, value)),
        2 => Just(Op::Begin),
        1 => Just(Op::Commit),
        2 => Just(Op::Rollback),
        1 => Just(Op::Clear),
    ]
}

/// A dict's observable state: entries in insertion order.
#[derive(Clone, Default, PartialEq, Debug)]
struct Model {
    entries: Vec<(usize, u32)>,
    saved: Option<Vec<(usize, u32)>>,
}

impl Model {
    fn apply(&mut self, op: &Op) {
        match *op {
            Op::Insert(key, value) => match self.entries.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = value,
                None => self.entries.push((key, value)),
            },
            Op::Begin => self.saved = Some(self.entries.clone()),
            Op::Commit => self.saved = None,
            Op::Rollback => {
                if let Some(saved) = self.saved.take() {
                    self.entries = saved;
                }
            }
            Op::Clear => {
                self.entries.clear();
                self.saved = None;
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn memo_matches_an_ordered_dict_with_transactions(ops in prop::collection::vec(op(), 0..64)) {
        let mut memo = Memo::new();
        let mut model = Model::default();
        for op in &ops {
            match *op {
                Op::Insert(key, value) => {
                    // A replaced entry reports the position the model keeps it at.
                    let position = model.entries.iter().position(|&(known, _)| known == key);
                    prop_assert_eq!(memo.insert(key, value), position);
                }
                Op::Begin => memo.begin(),
                Op::Commit => memo.commit(),
                Op::Rollback => memo.rollback(),
                Op::Clear => memo.clear(),
            }
            model.apply(op);
            let actual: Vec<(usize, u32)> = memo.iter().map(|(key, value)| (key, *value)).collect();
            prop_assert_eq!(&actual, &model.entries);
            prop_assert_eq!(memo.len(), model.entries.len());
            prop_assert_eq!(memo.is_empty(), model.entries.is_empty());
            for &(key, value) in &model.entries {
                prop_assert!(memo.contains(key));
                prop_assert_eq!(memo.get(key), Some(&value));
            }
        }
    }
}
