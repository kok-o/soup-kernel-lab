"""Semantic fixtures for the metadata optimization, independent of GPU math."""

import importlib.metadata as metadata
import pathlib
import sys
import zipfile

import pytest

from solution import metadata_exact


@pytest.fixture
def original():
    # The full suite may install the runtime before fixtures are constructed.
    # Compare both the semantics and stat counts against the real stdlib API.
    function = metadata.packages_distributions
    while getattr(function, "_setww_metadata_exact", False):
        function = function.__wrapped__
    return function


def dist(root, name, record=None, declared=None, egg=None):
    path = root / (name + (".egg-info" if egg is not None else ".dist-info"))
    path.mkdir()
    (path / "METADATA").write_text("Name: " + name + "\nVersion: 1\n")
    for filename, value in (("RECORD", record), ("top_level.txt", declared), ("SOURCES.txt", egg)):
        if value is not None:
            (path / filename).write_text(value)
    return metadata.PathDistribution(path)


def live(root, name):
    path = root / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text("")
    return path


def parity(monkeypatch, original, distributions, expected):
    monkeypatch.setattr(metadata, "distributions", lambda: iter(distributions))
    assert metadata_exact.packages_distributions(original) == original() == expected


def test_missing_declared_namespace_order(tmp_path, monkeypatch, original):
    missing = dist(tmp_path, "Missing", "ghost/a.py,,\nmissing.py,,\n")
    declared = dist(tmp_path, "Declared", "ghost/a.py,,\n", "declared\n")
    live(tmp_path, "namespace/first.py")
    live(tmp_path, "namespace/second.py")
    first = dist(
        tmp_path, "First", "namespace/missing.py,,\nnamespace/first.py,,\nother/missing.py,,\n"
    )
    second = dist(tmp_path, "Second", "namespace/second.py,,\n")
    parity(
        monkeypatch,
        original,
        [missing, declared, second, first, second],
        {"declared": ["Declared"], "namespace": ["Second", "First", "Second"]},
    )


def test_mutations_between_calls(tmp_path, monkeypatch, original):
    distribution = dist(tmp_path, "Mutable", "pkg/dead.py,,\npkg/live.py,,\n")
    path = live(tmp_path, "pkg/live.py")
    parity(monkeypatch, original, [distribution], {"pkg": ["Mutable"]})
    path.unlink()
    parity(monkeypatch, original, [distribution], {})
    (distribution._path / "RECORD").write_text("new.py,,\n")
    (distribution._path / "METADATA").write_text("Name: Renamed\n")
    live(tmp_path, "new.py")
    parity(monkeypatch, original, [distribution], {"new": ["Renamed"]})


@pytest.mark.parametrize("suffix", ["py", "pyc", "so", "pth", "dist-info"])
def test_top_level_module_inference(tmp_path, monkeypatch, original, suffix):
    name = "module." + suffix
    live(tmp_path, name)
    distribution = dist(tmp_path, "ModuleDistribution", name + ",,\n")
    monkeypatch.setattr(metadata, "distributions", lambda: iter([distribution]))
    assert metadata_exact.packages_distributions(original) == original()


def test_egg_empty_custom_and_subclass_fallback(tmp_path, monkeypatch, original):
    live(tmp_path, "egg.py")
    egg = dist(tmp_path, "Egg", egg="egg.py\nmissing.py\n")
    empty = dist(tmp_path, "Empty", "")

    class Custom(metadata.Distribution):
        def read_text(self, name):
            return "Name: Custom\n" if name == "METADATA" else None

        def locate_file(self, path):
            raise AssertionError("Do not reinterpret custom provider files")

        @property
        def files(self):
            return [metadata.PackagePath("virtual/module.py")]

    class Subclass(metadata.PathDistribution):
        @property
        def files(self):
            return [metadata.PackagePath("override/module.py")]

    parity(
        monkeypatch,
        original,
        [egg, empty, Custom(), Subclass(egg._path)],
        {"egg": ["Egg"], "virtual": ["Custom"], "override": ["Egg"]},
    )


def test_zip_provider(tmp_path, monkeypatch, original):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Zip.dist-info/METADATA", "Name: Zip\n")
        z.writestr("Zip.dist-info/RECORD", "zipns/missing.py,,\nzipns/live.py,,\n")
        z.writestr("zipns/live.py", "")
    with zipfile.ZipFile(archive) as z:
        distribution = metadata.PathDistribution(zipfile.Path(z, "Zip.dist-info/"))
        parity(monkeypatch, original, [distribution], {"zipns": ["Zip"]})


@pytest.mark.parametrize("invalid", ["pkg/a.py,,,extra\n", "pkg/a.py,,invalid\n", "\n"])
def test_malformed_rows_after_witness_keep_exception(tmp_path, monkeypatch, original, invalid):
    live(tmp_path, "pkg/a.py")
    distribution = dist(tmp_path, "Malformed", "pkg/a.py,,\n" + invalid)
    monkeypatch.setattr(metadata, "distributions", lambda: iter([distribution]))
    with pytest.raises(Exception) as expected:
        original()
    with pytest.raises(type(expected.value)) as actual:
        metadata_exact.packages_distributions(original)
    assert str(actual.value) == str(expected.value)


def test_hash_text_accepted_by_stdlib_is_not_rejected(tmp_path, monkeypatch, original):
    live(tmp_path, "pkg/a.py")
    distribution = dist(tmp_path, "HashText", "pkg/a.py,bad-hash,1\n")
    parity(monkeypatch, original, [distribution], {"pkg": ["HashText"]})


def test_existence_short_circuit_preserves_missing_names(tmp_path, monkeypatch, original):
    for index in range(5):
        live(tmp_path, f"present/{index}.py")
    distribution = dist(
        tmp_path,
        "Count",
        "present/absent.py,,\n"
        + "".join(f"present/{i}.py,,\n" for i in range(5))
        + "missing/0.py,,\nmissing/1.py,,\n",
    )
    monkeypatch.setattr(metadata, "distributions", lambda: iter([distribution]))
    exists = pathlib.Path.exists
    calls = []

    def counted(path):
        calls.append(path)
        return exists(path)

    monkeypatch.setattr(pathlib.Path, "exists", counted)
    expected = original()
    assert len(calls) == 8
    calls.clear()
    assert metadata_exact.packages_distributions(original) == expected == {"present": ["Count"]}
    assert len(calls) == 4


def test_install_idempotent_preserves_package_apis(monkeypatch, original):
    monkeypatch.setattr(metadata, "packages_distributions", original)
    before = {
        name: getattr(metadata, name)
        for name in ("distributions", "distribution", "version", "files", "requires")
    }
    descriptor = metadata.Distribution.files
    enabled = metadata_exact.install()
    assert enabled == (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 13))
    first = metadata.packages_distributions
    metadata_exact.install()
    assert metadata.packages_distributions is first
    assert all(getattr(metadata, name) is value for name, value in before.items())
    assert metadata.Distribution.files is descriptor
