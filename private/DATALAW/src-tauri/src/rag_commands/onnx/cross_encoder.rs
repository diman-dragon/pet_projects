use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::OnceLock;

// Using a static session to avoid re-loading model.
static SESSION: OnceLock<Session> = OnceLock::new();

pub struct CrossEncoder;

impl CrossEncoder {
    pub fn init(model_path: &str) -> Result<(), String> {
        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(4)
            .map_err(|e| e.to_string())?
            .commit_from_file(model_path)
            .map_err(|e| e.to_string())?;
        
        SESSION.set(session).map_err(|_| "Session already initialized".to_string())?;
        Ok(())
    }

    pub fn predict(_query: &str, _text: &str) -> Result<f32, String> {
        let _session = SESSION.get().ok_or("Model not initialized")?;
        Ok(0.5)
    }
}
