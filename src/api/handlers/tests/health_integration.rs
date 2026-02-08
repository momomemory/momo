#[cfg(test)]
mod integration {
    use crate::api::handlers::health::{compute_reranker_status, RerankerStatus};
    use crate::config::RerankerConfig;
    use axum::Json;

    #[test]
    fn test_compute_reranker_status_not_configured() {
        let status = compute_reranker_status::<()>(&None, &None);
        assert!(!status.enabled);
        assert_eq!(status.model, None);
        assert_eq!(status.status, "disabled");
    }

    #[test]
    fn test_compute_reranker_status_configured_no_provider() {
        let cfg = Some(RerankerConfig::default());
        let status = compute_reranker_status::<()>(&None, &cfg);
        assert!(!status.enabled);
        assert_eq!(status.model, Some(cfg.unwrap().model));
        assert_eq!(status.status, "error");
    }
}
