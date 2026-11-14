"""The image formats the build decodes are the ones it lists and documents.

Three lists must be one: the ``image`` crate features the workspace enables
(root ``Cargo.toml``, ``[workspace.dependencies]``), the committed fixtures
under ``tests/fixtures/formats/`` (``sample.<feature>``, one per feature), and
the table in ``docs/user-guide/concepts/sources.md``. Every fixture is decoded
through the pipeline, so removing a feature fails its decode here, and adding
one without a fixture and a row in the docs fails the set comparison.

The fixtures hold one 16x16 image (``fixtures/formats/generate.py``); what the
decode is compared with is that image, written out again below.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

from ._doc_tables import cell_code, table_with_header
from .conftest import plugin_required

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
_FIXTURES = Path(__file__).parent / "fixtures" / "formats"
_SOURCES_DOC = _PKG / "docs" / "user-guide" / "concepts" / "sources.md"


def _workspace_image() -> dict:
    manifest = tomllib.loads((_ROOT / "Cargo.toml").read_text())
    return manifest["workspace"]["dependencies"]["image"]


#: The features that are not a format (root ``Cargo.toml`` says why each is on).
_NOT_FORMATS = {"rayon"}


def _features() -> list[str]:
    """The formats the manifest enables, each by its feature name."""
    return sorted(set(_workspace_image()["features"]) - _NOT_FORMATS)


def _fixtures() -> list[str]:
    """The features with a committed file: what the decode tests run over, so
    a feature taken out of the manifest fails its file's decode."""
    return sorted(p.suffix[1:] for p in _FIXTURES.iterdir() if p.stem == "sample")


def _fixture(feature: str) -> bytes:
    return (_FIXTURES / f"sample.{feature}").read_bytes()


def _expected() -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:8, :8] = (200, 30, 30)
    img[:8, 8:] = (30, 200, 30)
    img[8:, :8] = (30, 30, 200)
    img[8:, 8:] = (220, 220, 60)
    return img


#: Largest difference from the fixture's image, in 8-bit steps. Zero unless
#: the format is lossy as written: JPEG, DXT1 (5:6:5 endpoints), and the two
#: float formats, which hold the image scaled to [0, 1] (Radiance stores an
#: 8-bit mantissa, OpenEXR as OpenCV writes it a half float).
_TOLERANCE = {"jpeg": 3, "dds": 8, "hdr": 2, "exr": 1}


@pytest.mark.structural
class TestTheManifestIsTheList:
    def test_the_codecs_are_named_not_defaulted(self) -> None:
        image = _workspace_image()
        assert image["default-features"] is False
        # `avif` is an encoder only (`ravif`, which builds `rav1e`), and no
        # sink writes AVIF.
        assert "avif" not in image["features"]
        assert _NOT_FORMATS <= set(image["features"])

    @pytest.mark.parametrize("crate", ["view-buffer", "polars-cv"])
    def test_each_crate_takes_the_workspace_entry(self, crate: str) -> None:
        # Features unify across the workspace: one crate naming `image` on its
        # own, with defaults, would compile every codec back in for both.
        manifest = tomllib.loads((_ROOT / crate / "Cargo.toml").read_text())
        assert manifest["dependencies"]["image"] == {"workspace": True}

    def test_every_feature_has_a_fixture_and_every_fixture_a_feature(self) -> None:
        assert _fixtures() == _features()

    def test_the_docs_list_exactly_the_features(self) -> None:
        rows = table_with_header(_SOURCES_DOC.read_text(), "Format", "`image` feature")
        assert rows is not None, "the image-formats table is missing"
        documented = sorted(cell_code(row[1]) or row[1] for row in rows)
        assert documented == _features()


def _decode(data: bytes) -> np.ndarray:
    df = pl.DataFrame({"img": [data]}, schema={"img": pl.Binary})
    out = df.select(
        out=pl.col("img").cv.pipe(Pipeline().source("image_bytes")).sink("tensor")
    )
    return numpy_from_struct(out["out"][0])


@plugin_required
@pytest.mark.parametrize("feature", _fixtures())
class TestEveryListedFormatDecodes:
    def test_the_pixels_are_the_fixtures(self, feature: str) -> None:
        got = _decode(_fixture(feature))
        assert got.shape[:2] == (16, 16)
        assert got.shape[2] in (3, 4)
        # To 8-bit steps, whatever the format stores: a float image is in
        # [0, 1], a 16-bit one (farbfeld) in [0, 65535].
        full_scale = 1.0 if got.dtype.kind == "f" else np.iinfo(got.dtype).max
        rgb = got[..., :3].astype(np.float64) * (255.0 / full_scale)
        worst = np.abs(rgb - _expected()).max()
        assert worst <= _TOLERANCE.get(feature, 0), worst

    def test_the_header_reads_without_a_decode(self, feature: str) -> None:
        df = pl.DataFrame({"img": [_fixture(feature)]}, schema={"img": pl.Binary})
        info = df.select(info=pl.col("img").cv.image_info())["info"][0]
        assert (info["width"], info["height"]) == (16, 16)
