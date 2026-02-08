use lru::LruCache;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Thread-safe LRU cache for storing rewritten queries
///
/// Uses Arc<Mutex<>> pattern for safe concurrent access across threads.
/// Cache keys are hashes of the original query string.
#[derive(Clone)]
pub struct QueryRewriteCache {
    cache: Arc<Mutex<LruCache<String, String>>>,
}

impl QueryRewriteCache {
    /// Create a new QueryRewriteCache with the specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of entries to store (LRU eviction applies)
    ///
    /// # Panics
    /// Panics if capacity is 0
    pub fn new(capacity: usize) -> Self {
        let cache = LruCache::new(NonZeroUsize::new(capacity).expect("Capacity must be non-zero"));
        Self {
            cache: Arc::new(Mutex::new(cache)),
        }
    }

    /// Retrieve a cached rewritten query
    ///
    /// # Arguments
    /// * `key` - The cache key (typically from generate_key())
    ///
    /// # Returns
    /// Some(rewritten_query) if found, None if not in cache
    pub fn get(&self, key: &str) -> Option<String> {
        let mut cache = self.cache.lock().unwrap();
        cache.get(key).cloned()
    }

    /// Store a rewritten query in the cache
    ///
    /// # Arguments
    /// * `key` - The cache key (typically from generate_key())
    /// * `value` - The rewritten query string
    ///
    /// If cache is at capacity, least recently used entry is evicted
    pub fn put(&self, key: String, value: String) {
        let mut cache = self.cache.lock().unwrap();
        cache.put(key, value);
    }

    /// Generate a stable hash key for a query string
    ///
    /// # Arguments
    /// * `query` - The original query to hash
    ///
    /// # Returns
    /// Hexadecimal string representation of the hash
    pub fn generate_key(&self, query: &str) -> String {
        let mut hasher = DefaultHasher::new();
        query.as_bytes().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_cache_hit_after_put() {
        let cache = QueryRewriteCache::new(10);
        let key = cache.generate_key("machine learning");
        let rewritten = "ML algorithms".to_string();

        cache.put(key.clone(), rewritten.clone());

        let result = cache.get(&key);
        assert_eq!(result, Some(rewritten));
    }

    #[test]
    fn test_cache_miss() {
        let cache = QueryRewriteCache::new(10);
        let result = cache.get("nonexistent_key");
        assert_eq!(result, None);
    }

    #[test]
    fn test_cache_capacity_enforcement() {
        let cache = QueryRewriteCache::new(2);

        let key1 = cache.generate_key("query1");
        let key2 = cache.generate_key("query2");
        let key3 = cache.generate_key("query3");

        cache.put(key1.clone(), "rewrite1".to_string());
        cache.put(key2.clone(), "rewrite2".to_string());
        cache.put(key3.clone(), "rewrite3".to_string());

        // key1 should be evicted (LRU)
        assert_eq!(cache.get(&key1), None);
        assert_eq!(cache.get(&key2), Some("rewrite2".to_string()));
        assert_eq!(cache.get(&key3), Some("rewrite3".to_string()));
    }

    #[test]
    fn test_key_generation_stability() {
        let cache = QueryRewriteCache::new(10);
        let query = "machine learning fundamentals";

        let key1 = cache.generate_key(query);
        let key2 = cache.generate_key(query);

        assert_eq!(key1, key2, "Same query should generate identical keys");
    }

    #[test]
    fn test_key_generation_uniqueness() {
        let cache = QueryRewriteCache::new(10);

        let key1 = cache.generate_key("query A");
        let key2 = cache.generate_key("query B");

        assert_ne!(
            key1, key2,
            "Different queries should generate different keys"
        );
    }

    #[test]
    fn test_concurrent_access() {
        let cache = QueryRewriteCache::new(100);
        let mut handles = vec![];

        // Spawn 10 threads, each writing and reading
        for i in 0..10 {
            let cache_clone = cache.clone();
            let handle = thread::spawn(move || {
                let query = format!("query_{}", i);
                let key = cache_clone.generate_key(&query);
                let value = format!("rewritten_{}", i);

                cache_clone.put(key.clone(), value.clone());

                // Read back immediately
                let result = cache_clone.get(&key);
                assert_eq!(result, Some(value));
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_lru_ordering() {
        let cache = QueryRewriteCache::new(3);

        let key1 = cache.generate_key("q1");
        let key2 = cache.generate_key("q2");
        let key3 = cache.generate_key("q3");
        let key4 = cache.generate_key("q4");

        cache.put(key1.clone(), "v1".to_string());
        cache.put(key2.clone(), "v2".to_string());
        cache.put(key3.clone(), "v3".to_string());

        // Access key1 to make it recently used
        let _ = cache.get(&key1);

        // Now add key4, which should evict key2 (least recently used)
        cache.put(key4.clone(), "v4".to_string());

        assert_eq!(cache.get(&key1), Some("v1".to_string()));
        assert_eq!(cache.get(&key2), None); // Evicted
        assert_eq!(cache.get(&key3), Some("v3".to_string()));
        assert_eq!(cache.get(&key4), Some("v4".to_string()));
    }
}
