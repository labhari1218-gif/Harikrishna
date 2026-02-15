"""
cache.py - SQLite-based disk cache for PV scores

Implements PVCache for caching premise validation scores with versioned keys
to prevent stale scores after verbalization or model parameter changes.

Cache key: (claim_key, evidence_hash, model_name, max_length, verbalizer_id)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

from .evidence import PVResult

logger = logging.getLogger(__name__)


class PVCache:
    """
    SQLite-based disk cache for PV scores with persistent connection.
    
    Cache key includes:
    - claim_key: Stable hash of claim text
    - evidence_hash: Full SHA256 hash of evidence text (64 chars)
    - model_name: HuggingFace model name
    - max_length: Tokenizer max_length parameter
    - verbalizer_id: Verbalization version
    
    This prevents serving stale scores after verbalization logic changes.
    
    Attributes:
        cache_dir: Directory for SQLite database
        db_path: Path to pv_cache_v2.db
        legacy_db_path: Optional path to pv_cache.db for backward-compatible reads
        conn: Persistent SQLite connection
        legacy_conn: Optional read-only connection to legacy cache
        read_only: If True, only reads from cache (no writes)
        hit_count: Number of cache hits
        miss_count: Number of cache misses
    """
    
    def __init__(self, cache_dir: str, read_only: bool = False, commit_buffer_size: int = 256):
        """
        Initialize PVCache with persistent SQLite connection and buffered commits.
        
        Args:
            cache_dir: Directory to store cache DB
            read_only: If True, do not write to cache
            commit_buffer_size: Number of inserts before auto-commit (default: 256)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "pv_cache_v2.db"
        self.legacy_db_path = self.cache_dir / "pv_cache.db"
        self.read_only = read_only
        
        self.hit_count = 0
        self.miss_count = 0
        
        # Persistent connection
        self.conn = None
        self.legacy_conn = None
        self.commit_buffer_size = commit_buffer_size
        self._pending_commits = 0  # Track uncommitted inserts
        
        # Initialize database
        self._init_db()
        
        if self.legacy_conn is not None:
            logger.info(f"PVCache initialized at {self.db_path} (legacy={self.legacy_db_path}, read_only={read_only})")
        else:
            logger.info(f"PVCache initialized at {self.db_path} (read_only={read_only})")
    
    def _init_db(self):
        """Create database tables and enable WAL mode for better concurrency."""
        self.conn = sqlite3.connect(self.db_path)
        
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        
        cursor = self.conn.cursor()
        
        # Main scores table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pv_scores (
                claim_key TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                max_length INTEGER NOT NULL,
                verbalizer_id TEXT NOT NULL,
                probs_json TEXT NOT NULL,
                rel REAL NOT NULL,
                pol REAL NOT NULL,
                PRIMARY KEY (claim_key, evidence_hash, model_name, max_length, verbalizer_id)
            )
        """)
        
        # Metadata table (for storing transformers version, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        self.conn.commit()

        # Optional legacy cache (read-only) for backward compatibility
        if self.legacy_db_path.exists():
            try:
                self.legacy_conn = sqlite3.connect(f"file:{self.legacy_db_path}?mode=ro", uri=True)
            except Exception as e:
                logger.warning(f"Failed to open legacy cache {self.legacy_db_path}: {e}")
                self.legacy_conn = None
    
    def flush(self):
        """Force commit any pending writes."""
        if self.conn and self._pending_commits > 0:
            self.conn.commit()
            self._pending_commits = 0
            logger.debug("Flushed pending cache commits")
    
    def close(self):
        """Close the persistent connection with WAL checkpoint."""
        if self.conn:
            try:
                # Flush any pending commits
                self.flush()
                # Checkpoint WAL to prevent unbounded growth
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                logger.warning(f"Error during cache close: {e}")
            finally:
                self.conn.close()
                self.conn = None
        if self.legacy_conn:
            try:
                self.legacy_conn.close()
            finally:
                self.legacy_conn = None
    
    def get(
        self,
        claim_key: str,
        evidence_hash: str,
        model_name: str,
        max_length: int,
        verbalizer_id: str,
        claim_id: Optional[str] = None
    ) -> Optional[PVResult]:
        """
        Retrieve cached PV result using persistent connection.
        
        Args:
            claim_key: Stable hash of claim text
            evidence_hash: Full SHA256 hash of evidence text (64 chars)
            model_name: Model name
            max_length: Max token length
            verbalizer_id: Verbalization version
            claim_id: Optional legacy claim identifier for backward-compatible reads
            
        Returns:
            PVResult if found, None otherwise
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT probs_json, rel, pol
            FROM pv_scores
            WHERE claim_key = ?
              AND evidence_hash = ?
              AND model_name = ?
              AND max_length = ?
              AND verbalizer_id = ?
        """, (claim_key, evidence_hash, model_name, max_length, verbalizer_id))

        row = cursor.fetchone()

        # Backward compatibility: check legacy cache by claim_id if available
        if row is None and claim_id and self.legacy_conn is not None:
            legacy_cursor = self.legacy_conn.cursor()
            legacy_cursor.execute("""
                SELECT probs_json, rel, pol
                FROM pv_scores
                WHERE claim_id = ?
                  AND evidence_hash = ?
                  AND model_name = ?
                  AND max_length = ?
                  AND verbalizer_id = ?
            """, (claim_id, evidence_hash, model_name, max_length, verbalizer_id))
            row = legacy_cursor.fetchone()

        if row is None:
            self.miss_count += 1
            return None

        self.hit_count += 1
        
        # Deserialize
        probs_json, rel, pol = row
        probs = json.loads(probs_json)
        
        return PVResult(
            p_entail=probs['entail'],
            p_contra=probs['contra'],
            p_neutral=probs['neutral'],
            rel=rel,
            pol=pol
        )
    
    def put(
        self,
        claim_key: str,
        evidence_hash: str,
        model_name: str,
        max_length: int,
        verbalizer_id: str,
        pv: PVResult
    ):
        """
        Store PV result in cache using persistent connection.
        
        Args:
            claim_key: Stable hash of claim text
            evidence_hash: Full SHA256 hash of evidence text (64 chars)
            model_name: Model name
            max_length: Max token length
            verbalizer_id: Verbalization version
            pv: PVResult to cache
        """
        if self.read_only:
            return
        
        cursor = self.conn.cursor()
        
        # Serialize probabilities
        probs_json = json.dumps({
            'entail': pv.p_entail,
            'contra': pv.p_contra,
            'neutral': pv.p_neutral
        })
        
        cursor.execute("""
            INSERT OR REPLACE INTO pv_scores
            (claim_key, evidence_hash, model_name, max_length, verbalizer_id, probs_json, rel, pol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (claim_key, evidence_hash, model_name, max_length, verbalizer_id,
              probs_json, pv.rel, pv.pol))
        
        # Buffered commits: only commit every N inserts
        self._pending_commits += 1
        if self._pending_commits >= self.commit_buffer_size:
            self.conn.commit()
            self._pending_commits = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with 'hits' and 'misses' counts
        """
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0.0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': hit_rate
        }
