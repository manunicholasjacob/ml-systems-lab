"""Allow ``python -m mlsyslab`` as an alias for the ``mlsys`` console script."""
from .cli import main

raise SystemExit(main())
