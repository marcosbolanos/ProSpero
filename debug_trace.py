import gzip
import json
import os
from pathlib import Path


class JsonlGzTraceWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(self.path, "at", encoding="utf-8")
        self.event_counts = {}
        self.n_events = 0

    def write(self, record):
        event = record.get("event", "unknown")
        self.event_counts[event] = self.event_counts.get(event, 0) + 1
        self.n_events += 1
        self.handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def summary(self):
        size_bytes = os.path.getsize(self.path) if self.path.exists() else 0
        return {
            "path": str(self.path),
            "format": "jsonl.gz",
            "n_events": int(self.n_events),
            "event_counts": self.event_counts,
            "size_bytes": int(size_bytes),
        }
