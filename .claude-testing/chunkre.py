#!/usr/bin/env python3
"""Print an anchored ctest -R alternation for the test names in a chunk file."""
import re
import sys

with open(sys.argv[1]) as fp:
    names = [line.strip() for line in fp if line.strip()]
print("^(" + "|".join(re.escape(n) for n in names) + ")$")
