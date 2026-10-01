import os
from datetime import datetime, timezone, timedelta

def current_week_file(out_dir,format):
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())
    if format == "csv":
        return os.path.join(out_dir, f"{week_start}.csv")
    elif format == "json":
        return os.path.join(out_dir, f"{week_start}.json")
    elif format == "jsonl":
        return os.path.join(out_dir, f"{week_start}.jsonl")
        