use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| "python".to_string());
    let output = Command::new(&python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .expect("无法运行 PYO3_PYTHON；请先激活包含 Python 的 Conda 环境");

    if !output.status.success() {
        panic!("无法从 Python 查询动态库目录");
    }

    let library_dir = String::from_utf8(output.stdout)
        .expect("Python 动态库目录不是 UTF-8")
        .trim()
        .to_string();

    if !library_dir.is_empty() && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{library_dir}");
    }
}
