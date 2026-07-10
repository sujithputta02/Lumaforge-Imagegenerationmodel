import sqlite3
import json
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="lumaforge.db"):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        """Creates a new SQLite database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes the database and creates the generations table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            prompt TEXT NOT NULL,
            expanded_prompt TEXT, -- JSON string
            negative_prompt TEXT,
            steps INTEGER,
            guidance_scale REAL,
            seed INTEGER,
            aspect_ratio TEXT,
            device TEXT,
            latency_sec REAL,
            memory_used_mb REAL,
            status TEXT,          -- 'completed', 'refused', 'error'
            image_path TEXT,      -- Path to the stored file on disk
            error_message TEXT,   -- Holds error logs if failed
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(query)
            print(f"[DatabaseManager] Database initialized at {os.path.abspath(self.db_path)}")
        finally:
            conn.close()

    def log_generation(self, session_id: str, prompt: str, status: str,
                       expanded_prompt: dict = None, negative_prompt: str = None,
                       steps: int = None, guidance_scale: float = None,
                       seed: int = -1, aspect_ratio: str = "1:1", device: str = "mps",
                       latency_sec: float = 0.0, memory_used_mb: float = 0.0,
                       image_path: str = None, error_message: str = None):
        """Logs a generation run (success or failure) to the SQLite database."""
        query = """
        INSERT OR REPLACE INTO generations (
            session_id, prompt, expanded_prompt, negative_prompt, steps,
            guidance_scale, seed, aspect_ratio, device, latency_sec,
            memory_used_mb, status, image_path, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        
        # Serialize dict to JSON string
        expanded_prompt_json = json.dumps(expanded_prompt) if expanded_prompt else None
        
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(query, (
                    session_id, prompt, expanded_prompt_json, negative_prompt, steps,
                    guidance_scale, seed, aspect_ratio, device, latency_sec,
                    memory_used_mb, status, image_path, error_message
                ))
            print(f"[DatabaseManager] Logged session {session_id} to database (status: {status})")
        except Exception as e:
            print(f"[DatabaseManager Error] Failed to log session {session_id}: {e}")
        finally:
            conn.close()

    def get_history(self, limit: int = 50) -> list:
        """Retrieves historical generation logs, returning a list of dicts."""
        query = "SELECT * FROM generations ORDER BY created_at DESC LIMIT ?;"
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            
            history = []
            for row in rows:
                item = dict(row)
                # Deserialize expanded prompt back to dict
                if item["expanded_prompt"]:
                    try:
                        item["expanded_prompt"] = json.loads(item["expanded_prompt"])
                    except Exception:
                        pass
                history.append(item)
            return history
        except Exception as e:
            print(f"[DatabaseManager Error] Failed to retrieve history: {e}")
            return []
        finally:
            conn.close()
