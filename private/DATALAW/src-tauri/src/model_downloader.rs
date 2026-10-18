use std::fs;
use std::path::PathBuf;
use std::io::copy;

pub fn get_model_dir() -> PathBuf {
    let mut path = std::env::current_dir().unwrap();
    path.push("models");
    path
}

pub fn ensure_reranker_model() -> Result<PathBuf, String> {
    let dir = get_model_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    
    let model_path = dir.join("reranker.onnx");
    
    if !model_path.exists() {
        println!("Downloading reranker model...");
        // Using a lightweight MiniLM cross-encoder for demonstration.
        // In production, point to your specific model URL.
        let url = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.onnx";
        let mut response = reqwest::blocking::get(url).map_err(|e| e.to_string())?;
        
        let mut file = fs::File::create(&model_path).map_err(|e| e.to_string())?;
        copy(&mut response, &mut file).map_err(|e| e.to_string())?;
    }
    
    Ok(model_path)
}
