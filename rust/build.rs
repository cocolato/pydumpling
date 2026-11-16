//! Lets the cdylib leave Python's symbols to the interpreter that loads it.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
