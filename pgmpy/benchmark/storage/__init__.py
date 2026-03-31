"""
Storage backend for benchmark results.

Supports multiple formats:
- In-memory (default)
- SQLite persistence
- JSON file storage
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ResultStore:
    """
    Store and retrieve benchmark results with multiple backends.
    
    Parameters
    ----------
    backend : str, default='memory'
        Storage backend: 'memory', 'sqlite', or 'json'
    path : str, optional
        File path for sqlite or json backend
    """

    def __init__(self, backend: str = "memory", path: Optional[str] = None):
        """Initialize result store."""
        if backend not in ["memory", "sqlite", "json"]:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend = backend
        self.path = path
        self._results: Dict[str, Any] = {}
        self._counter = 0

        if backend == "sqlite" and path:
            self._init_sqlite()
        elif backend == "json" and path:
            self._init_json()

        logger.info(f"✓ ResultStore initialized with backend={backend}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()

        # Create results table
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id TEXT UNIQUE,
                simulator TEXT,
                method TEXT,
                metrics TEXT,
                timestamp TEXT
            )
        """
        )
        self.conn.commit()
        logger.info(f"✓ SQLite database initialized at {self.path}")

    def _init_json(self) -> None:
        """Initialize JSON file storage."""
        self.json_path = Path(self.path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.json_path.exists():
            self.json_path.write_text(json.dumps({"results": []}))
            logger.info(f"✓ JSON storage initialized at {self.path}")

    def save(self, run_id: str, result: Dict[str, Any]) -> str:
        """
        Save a benchmark result.
        
        Parameters
        ----------
        run_id : str
            Unique identifier for the run
        result : dict
            Result data to store
            
        Returns
        -------
        str
            Saved run ID
        """
        try:
            if self.backend == "memory":
                self._results[run_id] = result
            elif self.backend == "sqlite":
                self._save_sqlite(run_id, result)
            elif self.backend == "json":
                self._save_json(run_id, result)

            logger.info(f"✓ Result saved: {run_id}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to save result {run_id}: {e}")
            raise

    def _save_sqlite(self, run_id: str, result: Dict) -> None:
        """Save to SQLite."""
        self.cursor.execute(
            """
            INSERT OR REPLACE INTO results (run_id, simulator, method, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                run_id,
                result.get("simulator", ""),
                result.get("method", ""),
                json.dumps(result.get("metrics", {})),
                result.get("timestamp", ""),
            ),
        )
        self.conn.commit()

    def _save_json(self, run_id: str, result: Dict) -> None:
        """Save to JSON file."""
        data = json.loads(self.json_path.read_text())
        data["results"].append({"run_id": run_id, **result})
        self.json_path.write_text(json.dumps(data, indent=2))

    def load(self, run_id: str) -> Optional[Dict]:
        """
        Load a benchmark result by ID.
        
        Parameters
        ----------
        run_id : str
            Run identifier
            
        Returns
        -------
        dict or None
            Result data if found
        """
        try:
            if self.backend == "memory":
                return self._results.get(run_id)
            elif self.backend == "sqlite":
                return self._load_sqlite(run_id)
            elif self.backend == "json":
                return self._load_json(run_id)
        except Exception as e:
            logger.error(f"Failed to load result {run_id}: {e}")
            return None

    def _load_sqlite(self, run_id: str) -> Optional[Dict]:
        """Load from SQLite."""
        self.cursor.execute("SELECT * FROM results WHERE run_id = ?", (run_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "run_id": row[1],
                "simulator": row[2],
                "method": row[3],
                "metrics": json.loads(row[4]),
                "timestamp": row[5],
            }
        return None

    def _load_json(self, run_id: str) -> Optional[Dict]:
        """Load from JSON file."""
        data = json.loads(self.json_path.read_text())
        for result in data.get("results", []):
            if result.get("run_id") == run_id:
                return result
        return None

    def list_all(self) -> List[str]:
        """
        List all stored run IDs.
        
        Returns
        -------
        list
            List of run identifiers
        """
        try:
            if self.backend == "memory":
                return list(self._results.keys())
            elif self.backend == "sqlite":
                self.cursor.execute("SELECT run_id FROM results")
                return [row[0] for row in self.cursor.fetchall()]
            elif self.backend == "json":
                data = json.loads(self.json_path.read_text())
                return [r.get("run_id") for r in data.get("results", [])]
        except Exception as e:
            logger.error(f"Failed to list results: {e}")
            return []

    def delete(self, run_id: str) -> bool:
        """
        Delete a stored result.
        
        Parameters
        ----------
        run_id : str
            Run identifier
            
        Returns
        -------
        bool
            True if deleted
        """
        try:
            if self.backend == "memory":
                if run_id in self._results:
                    del self._results[run_id]
                    return True
            elif self.backend == "sqlite":
                self.cursor.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
                self.conn.commit()
                return self.cursor.rowcount > 0
            elif self.backend == "json":
                data = json.loads(self.json_path.read_text())
                original_len = len(data["results"])
                data["results"] = [
                    r for r in data["results"] if r.get("run_id") != run_id
                ]
                self.json_path.write_text(json.dumps(data, indent=2))
                return len(data["results"]) < original_len
        except Exception as e:
            logger.error(f"Failed to delete result {run_id}: {e}")
            return False

    def clear(self) -> None:
        """Clear all stored results."""
        try:
            if self.backend == "memory":
                self._results.clear()
            elif self.backend == "sqlite":
                self.cursor.execute("DELETE FROM results")
                self.conn.commit()
            elif self.backend == "json":
                self.json_path.write_text(json.dumps({"results": []}))
            logger.info("✓ All results cleared")
        except Exception as e:
            logger.error(f"Failed to clear results: {e}")

    def close(self) -> None:
        """Close storage connection."""
        try:
            if self.backend == "sqlite" and hasattr(self, "conn"):
                self.conn.close()
                logger.info("✓ SQLite connection closed")
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = ["ResultStore"]
