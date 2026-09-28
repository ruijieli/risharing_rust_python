use anyhow::{Result, bail};

#[tokio::main]
async fn main() -> Result<()> {
    bail!(
        "Use `python training/train.py` or `python training/evaluate.py`. The Rust core is a Python extension, not a standalone application."
    )
}
