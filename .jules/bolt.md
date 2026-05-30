## 2025-05-15 - [@lru_cache on instance methods]
**Learning:** Using `@lru_cache` directly on an instance method includes `self` in the cache key. This creates a strong reference to the instance, which can lead to memory leaks as instances won't be garbage collected. It also means the cache is technically partitioned by instance identity.
**Action:** Use a closure-based cache inside `__init__` or a separate static/module-level cache if the instance doesn't need to be part of the key. Also, always return a `.copy()` of cached dictionaries to prevent mutation from affecting the cache.
