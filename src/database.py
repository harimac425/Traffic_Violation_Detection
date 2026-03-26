"""
Violation Database Module

SQLite-based storage for traffic violation records with frame snapshot evidence.
Stores violation type, timestamp, track ID, plate number, confidence, and 
a path to the captured evidence frame.
"""
import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


# Default paths
DEFAULT_DB_PATH = Path(__file__).parent.parent / "violations.db"
DEFAULT_EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"


@dataclass
class ViolationRecord:
    """Represents a stored violation record from the database"""
    id: int
    timestamp: str
    violation_type: str
    track_id: Optional[int]
    plate_number: Optional[str]
    confidence: float
    evidence_path: Optional[str]
    details: str
    llm_reasoning: Optional[str] = None


class ViolationDatabase:
    """
    SQLite database for storing traffic violation records.
    
    Features:
    - Auto-creates database and tables on first use
    - Saves violation frame snapshots as evidence images
    - Query violations by type, date range, or plate number
    - Export records to CSV
    - Provides summary statistics
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ViolationDatabase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: str = None, evidence_dir: str = None):
        if self._initialized:
            return
        
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self.evidence_dir = str(evidence_dir or DEFAULT_EVIDENCE_DIR)
        
        # Create evidence directory if it doesn't exist
        os.makedirs(self.evidence_dir, exist_ok=True)
        
        # Initialize database
        print(f"[Database] Active at: {self.db_path}")
        self._init_db()
        self._initialized = True
    
    def _init_db(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                track_id INTEGER,
                plate_number TEXT,
                confidence REAL NOT NULL,
                evidence_path TEXT,
                details TEXT DEFAULT '',
                llm_reasoning TEXT DEFAULT ''
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_violation_type 
            ON violations(violation_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON violations(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON violations(plate_number)
        """)
        
        conn.commit()
        conn.close()
        print(f"[Database] Initialized at: {self.db_path}")
    
    def save_evidence_frame(self, frame: np.ndarray, violation_type: str, 
                            track_id: Optional[int] = None) -> Optional[str]:
        """
        Save a frame as evidence image.
        
        Args:
            frame: The video frame (BGR numpy array)
            violation_type: Type of violation (used in filename)
            track_id: Optional track ID for the violating vehicle
            
        Returns:
            Path to the saved evidence image, or None if save failed
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            track_str = f"_T{track_id}" if track_id is not None else ""
            filename = f"{violation_type}{track_str}_{timestamp}.jpg"
            filepath = os.path.join(self.evidence_dir, filename)
            
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"[Database] Error saving evidence frame: {e}")
            return None
    
    def insert_violation(self, violation_type: str, confidence: float,
                         track_id: Optional[int] = None,
                         plate_number: Optional[str] = None,
                         evidence_path: Optional[str] = None,
                         details: str = "",
                         frame: np.ndarray = None,
                         llm_reasoning: str = "") -> int:
        """
        Insert a new violation record into the database.
        
        If a frame is provided and no evidence_path is given,
        the frame will be saved automatically as evidence.
        
        Args:
            violation_type: Type of violation (e.g., "no_helmet", "triple_riding")
            confidence: Detection confidence score (0.0 - 1.0)
            track_id: Optional vehicle/person track ID
            plate_number: Optional extracted license plate number
            evidence_path: Optional path to existing evidence image
            details: Additional violation details text
            frame: Optional video frame to save as evidence
            
        Returns:
            The row ID of the inserted record
        """
        # Auto-save evidence frame if provided
        if frame is not None and evidence_path is None:
            evidence_path = self.save_evidence_frame(frame, violation_type, track_id)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO violations 
            (timestamp, violation_type, track_id, plate_number, confidence, evidence_path, details, llm_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, violation_type, track_id, plate_number, 
              confidence, evidence_path, details, llm_reasoning))
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return row_id
    
    def get_violations(self, violation_type: str = None, 
                       limit: int = 100,
                       offset: int = 0,
                       date_from: str = None,
                       date_to: str = None,
                       plate_number: str = None) -> List[ViolationRecord]:
        """
        Query violation records with optional filters.
        
        Args:
            violation_type: Filter by violation type (None = all types)
            limit: Maximum number of records to return
            offset: Number of records to skip (for pagination)
            date_from: Start date filter (format: "YYYY-MM-DD")
            date_to: End date filter (format: "YYYY-MM-DD")
            plate_number: Filter by plate number (partial match)
            
        Returns:
            List of ViolationRecord objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM violations WHERE 1=1"
        params = []
        
        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)
        
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to + " 23:59:59")
        
        if plate_number:
            query += " AND plate_number LIKE ?"
            params.append(f"%{plate_number}%")
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            ViolationRecord(
                id=row[0],
                timestamp=row[1],
                violation_type=row[2],
                track_id=row[3],
                plate_number=row[4],
                confidence=row[5],
                evidence_path=row[6],
                details=row[7],
                llm_reasoning=row[8]
            )
            for row in rows
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get violation count statistics grouped by type.
        
        Returns:
            Dict mapping violation_type -> count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT violation_type, COUNT(*) 
            FROM violations 
            GROUP BY violation_type
            ORDER BY COUNT(*) DESC
        """)
        
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return stats
    
    def get_total_count(self) -> int:
        """Get total number of violation records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM violations")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def export_csv(self, output_path: str = None) -> str:
        """
        Export all violation records to a CSV file.
        
        Args:
            output_path: Output file path. Default: violations_export.csv in project root
            
        Returns:
            Path to the exported CSV file
        """
        import csv
        
        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(self.db_path), 
                f"violations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        
        records = self.get_violations(limit=100000)  # Get all records
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Timestamp', 'Violation Type', 'Track ID', 
                'Plate Number', 'Confidence', 'Evidence Path', 'Details'
            ])
            for r in records:
                writer.writerow([
                    r.id, r.timestamp, r.violation_type, r.track_id,
                    r.plate_number, f"{r.confidence:.2f}", r.evidence_path, r.details
                ])
        
        print(f"[Database] Exported {len(records)} records to: {output_path}")
        return output_path
    
    def delete_all(self):
        """Delete all violation records (use with caution!)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM violations")
        conn.commit()
        conn.close()
        print("[Database] All violation records deleted.")
    
    def close(self):
        """Clean up (no persistent connection to close, but included for interface consistency)"""
        pass
