"""Container health check: the status file must be fresh.

Usage: python -m kasegu.health [path] [max_age_s]
"""

import json
import sys
import time
from pathlib import Path


def check(path: Path, max_age_s: float) -> str | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return f"no status: {e}"
    age = time.time() - data["ts"] / 1000
    if age > max_age_s:
        return f"status is {age:.0f}s old"
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0] if argv else "data/status.json")
    err = check(path, float(argv[1]) if len(argv) > 1 else 60)
    print(err or "ok")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
