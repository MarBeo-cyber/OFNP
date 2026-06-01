"""
OFNP — Session Logger
Open Facial Neuroprosthesis Project

Implements Sec. 14 (Data Logging) requirements:
  - Stimulation intensity and timing
  - Trigger confidence
  - Skin impedance
  - Session duration
  - Safety events
  - Clinician notes

Privacy (Sec. 14.2):
  - All data local by default
  - Patient identifier is a hash (never name/DOB stored in log)
  - Logs exportable to anonymized CSV/JSON for research
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger("ofnp.logging")


@dataclass
class SessionMetadata:
    session_id: str
    patient_hash: str          # SHA-256 of patient_id — never store real ID
    clinician_id: str
    device_id: str
    start_timestamp: float
    phase: str                 # "simulation" / "bench" / "supervised"
    software_version: str = "0.1.0-MVP"
    firmware_version: str = "unknown"
    notes: str = ""


@dataclass
class LogEntry:
    timestamp: float
    entry_type: str
    data: dict
    session_id: str


class SessionLogger:
    """
    Structured session logger — writes JSONL (one JSON object per line).
    Format chosen for: streaming writes, easy parsing, clinical audit trail.
    """

    LOG_DIR = Path("/tmp/ofnp_sessions")

    def __init__(self, patient_id: str, clinician_id: str,
                 device_id: str = "OFNP-SIM-001",
                 phase: str = "simulation",
                 log_dir: Optional[Path] = None):
        self.log_dir = log_dir or self.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = str(uuid.uuid4())[:12]
        # Hash patient ID for privacy
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]

        self.metadata = SessionMetadata(
            session_id=self.session_id,
            patient_hash=patient_hash,
            clinician_id=clinician_id,
            device_id=device_id,
            start_timestamp=time.time(),
            phase=phase,
        )

        self._log_path = self.log_dir / f"session_{self.session_id}.jsonl"
        self._entry_count = 0

        self._write({"type": "session_start", "metadata": asdict(self.metadata)})
        logger.info("[Logger] Session %s started → %s", self.session_id, self._log_path)

    def log(self, entry_type: str, data: dict):
        """Log a structured event."""
        self._entry_count += 1
        self._write({
            "type":       entry_type,
            "session_id": self.session_id,
            "timestamp":  time.time(),
            "seq":        self._entry_count,
            **data,
        })

    def add_note(self, note: str):
        self.log("clinician_note", {"note": note})

    def close(self, notes: str = ""):
        self.log("session_end", {
            "duration_s":   round(time.time() - self.metadata.start_timestamp, 2),
            "total_entries": self._entry_count,
            "notes":         notes,
        })
        logger.info("[Logger] Session %s closed. %d entries → %s",
                    self.session_id, self._entry_count, self._log_path)

    def export_csv(self, output_path: Optional[Path] = None) -> Path:
        """Export session log to CSV (stimulation events only)."""
        import csv
        out = output_path or self.log_dir / f"session_{self.session_id}_export.csv"

        rows = []
        with open(self._log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") in ("stimulation_command", "expression_estimate"):
                        rows.append({
                            "timestamp":   entry.get("timestamp"),
                            "type":        entry.get("type"),
                            "expression":  entry.get("expression", ""),
                            "confidence":  entry.get("confidence", ""),
                            "intensity":   entry.get("intensity", ""),
                            "target":      entry.get("target_muscle", ""),
                            "latency_ms":  entry.get("pipeline_latency_ms", ""),
                        })
                except json.JSONDecodeError:
                    pass

        with open(out, "w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info("[Logger] CSV exported → %s (%d rows)", out, len(rows))
        return out

    def _write(self, data: dict):
        with open(self._log_path, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def entry_count(self) -> int:
        return self._entry_count
