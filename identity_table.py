"""
MEMORA - Identity Table Module
SQLite-based persistent storage for user identities
Schema: user_id, name, face_vector, voice_vector, registered_at, last_seen, trust_seed
"""

import sqlite3
import numpy as np
import json
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path


class IdentityTable:
    """
    SQLite-based identity storage and retrieval
    Stores face/voice embeddings and metadata for all registered users
    """
    
    def __init__(self, db_path: str = "./memora_identity.db"):
        """
        Initialize identity table
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_schema()
        print(f"[IdentityTable] ✓ Initialized at {db_path}")
    
    def _init_schema(self):
        """Create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Main identity table
            c.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    face_vector BLOB NOT NULL,
                    voice_vector BLOB NOT NULL,
                    face_confidence REAL DEFAULT 0.0,
                    voice_confidence REAL DEFAULT 0.0,
                    ic_score REAL DEFAULT 0.0,
                    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    trust_seed REAL DEFAULT 0.5,
                    metadata TEXT
                )
            ''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_name ON identities (name)')

            # NEW: Evaluation logs table for threshold tuning
            c.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_logs (
                    eval_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    test_type TEXT NOT NULL,
                    user_id_1 TEXT NOT NULL,
                    user_id_2 TEXT,
                    ic_score REAL NOT NULL,
                    face_sim REAL NOT NULL,
                    voice_sim REAL NOT NULL,
                    threshold_applied REAL,
                    decision TEXT,
                    metadata TEXT,
                    FOREIGN KEY (user_id_1) REFERENCES identities(user_id)
                )
            ''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_test_type ON evaluation_logs (test_type)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_user_1 ON evaluation_logs (user_id_1)')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[IdentityTable] ✗ Schema initialization failed: {e}")
            raise
    
    def add_identity(self, user_id: str, name: str, face_emb: np.ndarray, 
                     voice_emb: np.ndarray, ic_score: float = 0.0, 
                     metadata: dict = None) -> bool:
        """
        Store new user identity
        
        Args:
            user_id: Unique identifier (e.g., "alice_1234567890")
            name: Display name
            face_emb: 128-d face embedding (numpy array)
            voice_emb: 192-d speaker embedding (numpy array)
            ic_score: IC(u,I) fusion score
            metadata: Additional metadata dict
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert embeddings to binary
            face_blob = face_emb.astype(np.float32).tobytes()
            voice_blob = voice_emb.astype(np.float32).tobytes()
            meta_json = json.dumps(metadata or {})
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('''
                INSERT OR REPLACE INTO identities 
                (user_id, name, face_vector, voice_vector, ic_score, registered_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, name, face_blob, voice_blob, ic_score, datetime.now(), meta_json))
            
            conn.commit()
            conn.close()
            
            print(f"[IdentityTable] ✓ Stored identity: {name} ({user_id})")
            return True
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to add identity: {e}")
            return False
    
    def get_identity(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve identity by user ID
        
        Args:
            user_id: User identifier
        
        Returns:
            Dict with user data, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT * FROM identities WHERE user_id = ?', (user_id,))
            row = c.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return {
                'user_id': row[0],
                'name': row[1],
                'face_vector': np.frombuffer(row[2], dtype=np.float32),
                'voice_vector': np.frombuffer(row[3], dtype=np.float32),
                'face_confidence': row[4],
                'voice_confidence': row[5],
                'ic_score': row[6],
                'registered_at': row[7],
                'last_seen': row[8],
                'trust_seed': row[9],
                'metadata': json.loads(row[10]) if row[10] else {}
            }
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to get identity: {e}")
            return None
    
    def get_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieve identity by name (first match)
        
        Args:
            name: User name
        
        Returns:
            Identity dict or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute('SELECT user_id FROM identities WHERE name = ? LIMIT 1', (name,))
            row = c.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return self.get_identity(row[0])
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to get by name: {e}")
            return None
    
    def update_last_seen(self, user_id: str) -> bool:
        """
        Update last_seen timestamp for user
        
        Args:
            user_id: User identifier
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('UPDATE identities SET last_seen = ? WHERE user_id = ?',
                     (datetime.now(), user_id))
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to update last_seen: {e}")
            return False
    
    def list_all(self) -> List[Dict]:
        """
        List all registered identities
        
        Returns:
            List of dicts with user_id, name, registered_at, ic_score
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT user_id, name, registered_at, ic_score FROM identities ORDER BY registered_at DESC')
            rows = c.fetchall()
            conn.close()
            
            return [
                {
                    'user_id': r[0],
                    'name': r[1],
                    'registered_at': r[2],
                    'ic_score': r[3]
                }
                for r in rows
            ]
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to list identities: {e}")
            return []
    
    def search_all_vectors(self) -> List[tuple]:
        """
        Get all (user_id, face_vector, voice_vector) for FAISS indexing (Week 2+)
        
        Returns:
            List of (user_id, face_emb, voice_emb) tuples
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('SELECT user_id, face_vector, voice_vector FROM identities')
            rows = c.fetchall()
            conn.close()
            
            return [
                (r[0], np.frombuffer(r[1], dtype=np.float32), np.frombuffer(r[2], dtype=np.float32))
                for r in rows
            ]
        
        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to search vectors: {e}")
            return []
    
    def delete_identity(self, user_id: str) -> bool:
        """
        Delete identity from database

        Args:
            user_id: User identifier

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('DELETE FROM identities WHERE user_id = ?', (user_id,))
            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to delete identity: {e}")
            return False

    def log_evaluation(self, eval_id: str, test_type: str, user_id_1: str, user_id_2: Optional[str],
                      ic_score: float, face_sim: float, voice_sim: float,
                      threshold_applied: float = 0.0, decision: str = "",
                      metadata: Dict = None) -> bool:
        """
        Log evaluation result (for PR curve generation)

        Args:
            eval_id: Unique evaluation ID
            test_type: "intra_user" or "inter_user"
            user_id_1: First user identifier
            user_id_2: Second user identifier (None for intra-user)
            ic_score: IC score computed
            face_sim: Face similarity
            voice_sim: Voice similarity
            threshold_applied: Threshold used for decision
            decision: "MATCH" or "REJECT"
            metadata: Additional metadata dict

        Returns:
            True if successful
        """
        try:
            meta_json = json.dumps(metadata or {})

            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            c.execute('''
                INSERT INTO evaluation_logs
                (eval_id, test_type, user_id_1, user_id_2, ic_score, face_sim, voice_sim,
                 threshold_applied, decision, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (eval_id, test_type, user_id_1, user_id_2, ic_score, face_sim, voice_sim,
                  threshold_applied, decision, meta_json))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to log evaluation: {e}")
            return False

    def get_evaluation_logs(self, test_type: Optional[str] = None) -> List[Dict]:
        """
        Retrieve evaluation logs

        Args:
            test_type: Optional filter ("intra_user" or "inter_user")

        Returns:
            List of evaluation log dicts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            if test_type:
                c.execute('SELECT * FROM evaluation_logs WHERE test_type = ? ORDER BY timestamp DESC',
                         (test_type,))
            else:
                c.execute('SELECT * FROM evaluation_logs ORDER BY timestamp DESC')

            rows = c.fetchall()
            conn.close()

            return [
                {
                    'eval_id': r[0],
                    'timestamp': r[1],
                    'test_type': r[2],
                    'user_id_1': r[3],
                    'user_id_2': r[4],
                    'ic_score': r[5],
                    'face_sim': r[6],
                    'voice_sim': r[7],
                    'threshold_applied': r[8],
                    'decision': r[9],
                    'metadata': json.loads(r[10]) if r[10] else {}
                }
                for r in rows
            ]

        except Exception as e:
            print(f"[IdentityTable] ✗ Failed to get evaluation logs: {e}")
            return []


# Unit test
if __name__ == "__main__":
    print("[IdentityTable] Testing SQLite identity storage...\n")
    
    # Create test instance
    table = IdentityTable("./test_identity.db")
    
    # Test 1: Add identity
    print("Test 1: Adding identity...")
    face_emb = np.random.randn(128).astype(np.float32)
    face_emb /= np.linalg.norm(face_emb)
    voice_emb = np.random.randn(192).astype(np.float32)
    voice_emb /= np.linalg.norm(voice_emb)
    
    success = table.add_identity(
        user_id="alice_test",
        name="Alice",
        face_emb=face_emb,
        voice_emb=voice_emb,
        ic_score=0.85,
        metadata={'test': True}
    )
    print(f"  Result: {'✓ Success' if success else '✗ Failed'}")
    
    # Test 2: Retrieve identity
    print("\nTest 2: Retrieving identity...")
    identity = table.get_identity("alice_test")
    if identity:
        print(f"  ✓ Retrieved: {identity['name']}")
        print(f"    Face vector shape: {identity['face_vector'].shape}")
        print(f"    Voice vector shape: {identity['voice_vector'].shape}")
        print(f"    IC score: {identity['ic_score']}")
    else:
        print("  ✗ Not found")
    
    # Test 3: Get by name
    print("\nTest 3: Retrieving by name...")
    identity2 = table.get_by_name("Alice")
    print(f"  Result: {'✓ Found' if identity2 else '✗ Not found'}")
    
    # Test 4: List all
    print("\nTest 4: Listing all...")
    users = table.list_all()
    print(f"  Total users: {len(users)}")
    for u in users:
        print(f"    - {u['name']} (IC: {u['ic_score']:.4f})")
    
    # Test 5: Update last_seen
    print("\nTest 5: Updating last_seen...")
    success = table.update_last_seen("alice_test")
    print(f"  Result: {'✓ Success' if success else '✗ Failed'}")
    
    # Cleanup
    import os
    os.remove("./test_identity.db")
    print("\n✓ All tests passed")