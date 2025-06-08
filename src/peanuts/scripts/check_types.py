#!/usr/bin/env python3
"""Type checking script for peanuts project."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run ty type checker on peanuts package only."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    peanuts_dir = project_root / "peanuts"

    # Run ty check on peanuts directory
    cmd = ["uv", "run", "ty", "check", str(peanuts_dir)]

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running type checker: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
