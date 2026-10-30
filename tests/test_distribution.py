"""
Check package archive contents when PHSI_DIST_DIR is set.
"""

import os
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("PHSI_DIST_DIR"),
    reason="Set PHSI_DIST_DIR after building distributions.",
)
def test_distribution_keeps_private_instructions_out_and_examples_lowercase():
    license_text = (Path(__file__).parents[1] / "LICENSE").read_bytes()
    for notice in (
        b"Copyright (c) 2026 SINTEF and contributors",
        b"Copyright (c) 2025 SINTEF",
        b"Copyright (c) [2022] [SINTEF Digital]",
    ):
        assert notice in license_text
    directory = Path(os.environ["PHSI_DIST_DIR"])
    sources = list(directory.glob("phsi_models-*.tar.gz"))
    wheels = list(directory.glob("phsi_models-*.whl"))
    assert sources and wheels
    for source in sources:
        with tarfile.open(source) as archive:
            license_name = next(
                name for name in archive.getnames() if name.endswith("/LICENSE")
            )
            assert archive.extractfile(license_name).read() == license_text
            names = {
                name.split("/", 1)[1] for name in archive.getnames() if "/" in name
            }
        assert not any(".ipynb_checkpoints" in name.split("/") for name in names)
        assert "src/phsi/__init__.py" in names
        assert "LICENSE" in names
        assert "LICENSE-MIT.proposed" not in names
        assert "AGENTS.md" in names
        assert "docs/public-agent-guide.md" not in names
        assert "RELEASE_POLICY.md" not in names
        assert "docs/verification.md" not in names
        with tarfile.open(source) as guide_archive:
            guide_name = next(
                name for name in guide_archive.getnames() if name.endswith("/AGENTS.md")
            )
            guide = guide_archive.extractfile(guide_name).read().decode()
        assert guide.startswith("# Working with the PHSI package")
        assert "Private development instructions" not in guide
        assert {"examples/two_mass.ipynb", "examples/electric_circuit.ipynb"} <= names
        assert not any(
            name.startswith(("Examples/", ".local/", "examples/output/"))
            for name in names
        )
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata = archive.read(
                next(name for name in names if name.endswith("/METADATA"))
            ).decode()
            assert "Name: phsi-models\n" in metadata
            assert "phsi/__init__.py" in names
            assert "License-Expression: MIT" in metadata
            assert "License-File: LICENSE\n" in metadata
            assert metadata.count("License-File: ") == 1
            license_name = next(
                name for name in names if name.endswith(".dist-info/licenses/LICENSE")
            )
            assert archive.read(license_name) == license_text
            assert not any("LICENSE-MIT.proposed" in name for name in names)
        assert not any(
            name.endswith("/AGENTS.md") or name == "AGENTS.md" or ".local/" in name
            for name in names
        )
