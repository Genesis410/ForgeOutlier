"""Command-line entrypoint for python -m forgeoutlier."""

import sys
from forgeoutlier.cli.commands import main

if __name__ == "__main__":
    sys.exit(main())
