use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;

#[cfg(target_os = "windows")]
use std::process::Command;

const COMPACT_KEY_PREFIX: &str = "DL1-";
const CHARSET: &str = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
const EPOCH_DATE_MS: u64 = 1776038400000; // 2026-04-10 00:00:00 UTC

// v1: data(2) + hmac(8)  = 10 bytes → 16 chars → DL1-XXXX-XXXX-XXXX-XXXX
// v2: data(2) + nonce(3) + hmac(8) = 13 bytes → 21 chars padded to 24 → DL1-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX
const V1_RAW_LEN: usize = 16;
const V2_RAW_LEN: usize = 24;

fn master_key() -> &'static str {
    option_env!("Dorlane_LICENSE_MASTER_KEY").unwrap_or("")
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineFingerprintPayload {
    pub fingerprint: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IssuedLicenseTokenPayload {
    pub key: String,
    pub tier: String,
    pub expires_at: Option<String>,
    pub issued_at: String,
}

fn tier_from_val(val: u8) -> Option<&'static str> {
    match val {
        1 => Some("basic"),
        2 => Some("pro"),
        3 => Some("enterprise"),
        _ => None,
    }
}

fn decode_base32(input: &str) -> Result<Vec<u8>, String> {
    let mut bits = 0u32;
    let mut value = 0u32;
    let mut output = Vec::new();

    for c in input.chars() {
        if c == '-' {
            continue;
        }
        let idx = CHARSET
            .find(c)
            .ok_or_else(|| format!("Invalid character in key: {}", c))? as u32;
        value = (value << 5) | idx;
        bits += 5;

        if bits >= 8 {
            output.push((value >> (bits - 8)) as u8);
            bits -= 8;
            value &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

/// Count raw (non-dash) characters in key after prefix
fn raw_len(key: &str) -> usize {
    key.chars().filter(|c| *c != '-').count()
}

#[tauri::command]
pub fn verify_master_key(input: String) -> bool {
    !master_key().is_empty() && input.trim() == master_key()
}

#[tauri::command]
pub fn verify_license_key(license_key: String) -> Result<IssuedLicenseTokenPayload, String> {
    let normalized = license_key.trim().to_uppercase();

    if !normalized.starts_with(COMPACT_KEY_PREFIX) {
        return Err("Неверный формат ключа".into());
    }

    let after_prefix = &normalized[COMPACT_KEY_PREFIX.len()..];
    let chars_no_dash = raw_len(after_prefix);

    // Detect version by key length
    if chars_no_dash == V1_RAW_LEN {
        verify_v1(&normalized, after_prefix)
    } else if chars_no_dash == V2_RAW_LEN {
        verify_v2(&normalized, after_prefix)
    } else {
        Err(format!(
            "Неверная длина ключа: {} символов (ожидается {} или {})",
            chars_no_dash, V1_RAW_LEN, V2_RAW_LEN
        ))
    }
}

/// v1: data(2) + hmac(8) = 10 bytes
fn verify_v1(normalized: &str, raw_key: &str) -> Result<IssuedLicenseTokenPayload, String> {
    let decoded = decode_base32(raw_key)?;

    if decoded.len() < 10 {
        return Err("Ключ v1 слишком короткий".into());
    }

    let data = &decoded[0..2];
    let signature = &decoded[2..10];

    let mut mac = Hmac::<Sha256>::new_from_slice(master_key().as_bytes())
        .map_err(|_| "Internal error: HMAC init failed")?;
    mac.update(data);
    let expected = mac.finalize().into_bytes();

    if signature != &expected[..8] {
        return Err("Неверная подпись лицензионного ключа".into());
    }

    parse_payload(normalized, data)
}

/// v2: data(2) + nonce(3) + hmac(8) = 13 bytes
fn verify_v2(normalized: &str, raw_key: &str) -> Result<IssuedLicenseTokenPayload, String> {
    let decoded = decode_base32(raw_key)?;

    if decoded.len() < 13 {
        return Err("Ключ v2 слишком короткий".into());
    }

    let data = &decoded[0..2];
    let nonce = &decoded[2..5];
    let signature = &decoded[5..13];

    let mut mac = Hmac::<Sha256>::new_from_slice(master_key().as_bytes())
        .map_err(|_| "Internal error: HMAC init failed")?;
    mac.update(data);
    mac.update(nonce);
    let expected = mac.finalize().into_bytes();

    if signature != &expected[..8] {
        return Err("Неверная подпись лицензионного ключа".into());
    }

    parse_payload(normalized, data)
}

fn parse_payload(normalized: &str, data: &[u8]) -> Result<IssuedLicenseTokenPayload, String> {
    let tier_val = (data[0] >> 4) & 0x0F;
    let expiry_days = (((data[0] & 0x0F) as u32) << 8) | (data[1] as u32);

    let tier = tier_from_val(tier_val).ok_or("Неизвестный уровень лицензии")?;

    let expires_at = if expiry_days == 0 {
        None
    } else {
        let timestamp_ms = EPOCH_DATE_MS + (expiry_days as u64) * 86_400_000;
        Some(iso_from_unix_ms(timestamp_ms)?)
    };

    Ok(IssuedLicenseTokenPayload {
        key: normalized.to_string(),
        tier: tier.to_string(),
        expires_at,
        issued_at: iso_from_unix_ms(EPOCH_DATE_MS)?,
    })
}

fn iso_from_unix_ms(timestamp_ms: u64) -> Result<String, String> {
    let seconds = (timestamp_ms / 1000) as i64;
    let nanos = ((timestamp_ms % 1000) * 1_000_000) as u32;
    let datetime = chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanos)
        .ok_or("Неверная дата лицензии".to_string())?;
    Ok(datetime.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

#[cfg(target_os = "windows")]
fn collect_sources() -> Vec<(String, String)> {
    let mut result = Vec::new();

    let env_candidates = [
        ("computer_name", env::var("COMPUTERNAME")),
        ("processor_identifier", env::var("PROCESSOR_IDENTIFIER")),
        ("user_domain", env::var("USERDOMAIN")),
        ("home_drive", env::var("HOMEDRIVE")),
    ];

    for (label, value) in &env_candidates {
        if let Ok(v) = value {
            let trimmed = v.trim().to_string();
            if !trimmed.is_empty() {
                result.push((label.to_string(), trimmed));
            }
        }
    }

    let guid_output = Command::new("reg")
        .args([
            "query",
            r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Cryptography",
            "/v",
            "MachineGuid",
        ])
        .output();

    if let Ok(out) = guid_output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            if line.contains("MachineGuid") {
                let parts: Vec<&str> = line.splitn(3, "    ").collect();
                if let Some(guid) = parts.last() {
                    let guid = guid.trim().to_string();
                    if !guid.is_empty() {
                        result.push(("machine_guid".to_string(), guid));
                    }
                }
            }
        }
    }

    result
}

#[cfg(target_os = "macos")]
fn collect_sources() -> Vec<(String, String)> {
    let mut result = Vec::new();
    let hw_output = std::process::Command::new("ioreg")
        .args(["-rd1", "-c", "IOPlatformExpertDevice"])
        .output();

    if let Ok(out) = hw_output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        for line in stdout.lines() {
            if line.contains("IOPlatformUUID") {
                if let Some(uuid) = line.split('"').nth(3) {
                    result.push(("hw_uuid".to_string(), uuid.to_string()));
                }
            }
            if line.contains("IOPlatformSerialNumber") {
                if let Some(serial) = line.split('"').nth(3) {
                    result.push(("serial".to_string(), serial.to_string()));
                }
            }
        }
    }

    if result.is_empty() {
        if let Ok(hostname) = env::var("HOSTNAME").or_else(|_| {
            std::process::Command::new("hostname")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .map_err(|_| env::VarError::NotPresent)
        }) {
            result.push(("hostname".to_string(), hostname));
        }
    }

    result
}

#[cfg(target_os = "linux")]
fn collect_sources() -> Vec<(String, String)> {
    let mut result = Vec::new();

    if let Ok(id) = std::fs::read_to_string("/etc/machine-id") {
        let id = id.trim().to_string();
        if !id.is_empty() {
            result.push(("machine_id".to_string(), id));
        }
    }

    if result.is_empty() {
        if let Ok(id) = std::fs::read_to_string("/var/lib/dbus/machine-id") {
            let id = id.trim().to_string();
            if !id.is_empty() {
                result.push(("dbus_machine_id".to_string(), id));
            }
        }
    }

    if result.is_empty() {
        if let Ok(out) = std::process::Command::new("hostname").output() {
            let hostname = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !hostname.is_empty() {
                result.push(("hostname".to_string(), hostname));
            }
        }
    }

    result
}

#[tauri::command]
pub fn get_machine_fingerprint() -> MachineFingerprintPayload {
    let mut hasher = Sha256::new();
    let mut sources = Vec::new();
    let candidates = collect_sources();

    for (label, value) in &candidates {
        hasher.update(label.as_bytes());
        hasher.update(b":");
        hasher.update(value.as_bytes());
        hasher.update(b"|");
        sources.push(label.clone());
    }

    if candidates.is_empty() {
        hasher.update(b"fallback:unknown");
        sources.push("fallback".to_string());
    }

    let fingerprint = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<String>();

    MachineFingerprintPayload { fingerprint, sources }
}
