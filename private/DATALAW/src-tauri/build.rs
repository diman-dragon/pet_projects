use std::{env, path::PathBuf};

fn load_master_key() -> Result<String, String> {
    // Primary: read from environment variable (SECURE - never stored on disk)
    if let Ok(value) = env::var("Dorlane_LICENSE_MASTER_KEY") {
        let trimmed = value.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    // Development fallback: check .secrets file (should be in .gitignore and never committed)
    let workspace_root = env::var("CARGO_MANIFEST_DIR")
        .ok()
        .and_then(|p| PathBuf::from(p).parent().map(|p| p.to_path_buf()))
        .ok_or_else(|| "Failed to resolve workspace root".to_string())?;

    let secret_path = workspace_root.join(".secrets").join("license-master.key");

    if secret_path.exists() {
        let fs = std::fs::read_to_string(&secret_path)
            .map_err(|_| format!("Failed to read license key file: {}", secret_path.display()))?;
        let trimmed = fs.trim().to_string();
        if !trimmed.is_empty() {
            println!("cargo:warning=Using license key from file (set Dorlane_LICENSE_MASTER_KEY env var for production)");
            return Ok(trimmed);
        }
    }

    Err("License master key is required. Set Dorlane_LICENSE_MASTER_KEY environment variable.".to_string())
}

fn main() {
    println!("cargo:rerun-if-env-changed=Dorlane_LICENSE_MASTER_KEY");
    
    // Trigger rebuild if .secrets file changes
    if let Ok(dir) = env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(dir).parent().unwrap().join(".secrets").join("license-master.key");
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    }

    let master_key = load_master_key().expect(
        "License secret is required. Set Dorlane_LICENSE_MASTER_KEY or create .secrets/license-master.key",
    );
    println!("cargo:rustc-env=Dorlane_LICENSE_MASTER_KEY={master_key}");

    tauri_build::build()
}
