"""Exact grouped existence checks for CPython 3.13 package-name discovery.

TRL imports this mapping. Keep live-file discovery, metadata and list order;
only stop stat-ing a top-level name once one existing file witnesses it.
No cache, installed-package edits, or changes to Distribution.files/check APIs.
Concurrent filesystem mutation during one scan is outside this optimization.
Idea/proof: team's research_notes/Ускорение Soup после INT8/metadata_exact_probe.py.
"""

import collections
import csv
import importlib.metadata as metadata
import sys
from functools import wraps


def _record_path(name, hash=None, size_str=None):
    path = metadata.PackagePath(name)
    # Parse even rows whose package was already found: malformed rows must
    # still trigger the original API's exception, not disappear via early exit.
    if hash:
        metadata.FileHash(hash)
    if size_str:
        int(size_str)
    return path


def packages_distributions(original):
    try:
        result = collections.defaultdict(list)
        for dist in metadata.distributions():
            names = metadata._top_level_declared(dist)
            if not names:
                record = (
                    dist.read_text("RECORD") if type(dist) is metadata.PathDistribution else None
                )
                if not record:
                    names = metadata._top_level_inferred(dist)
                else:
                    found = set()
                    for row in csv.reader(record.splitlines()):
                        path = _record_path(*row)
                        name = metadata._get_toplevel_name(path)
                        if (
                            "." not in name
                            and name not in found
                            and dist.locate_file(path).exists()
                        ):
                            found.add(name)
                    names = found
            for name in names:
                result[name].append(dist.metadata["Name"])
        return dict(result)
    except Exception:
        return original()


def install():
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        return False
    original = metadata.packages_distributions
    if getattr(original, "_setww_metadata_exact", False):
        return True
    if not all(
        hasattr(metadata, name)
        for name in ("_top_level_declared", "_top_level_inferred", "_get_toplevel_name")
    ):
        return False

    @wraps(original)
    def replacement():
        return packages_distributions(original)

    replacement._setww_metadata_exact = True
    metadata.packages_distributions = replacement
    return True
