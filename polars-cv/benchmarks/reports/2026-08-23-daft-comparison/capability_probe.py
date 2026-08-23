"""
Capability probe: what Daft and polars-cv can each actually express.

The throughput numbers in `README.md` only cover operations both engines can
run. This script covers the other half of the comparison — what each engine
*can* do — and it is committed so every API claim in the report can be
re-checked against a future Daft release instead of being taken on trust.

Each probe is a small, self-contained fact:

- ``op_surface``       — how many image operations each side exposes.
- ``luma``             — the grayscale convention each engine uses.
- ``per_row_params``   — whether operation parameters may vary per row.
- ``type_system``      — what Daft's image/tensor columns support.

Several Daft probes crash the Rust worker rather than raising (see
`README.md` §5), so anything that can abort the interpreter runs in a
subprocess and is reported by exit status. Run it with::

    uv run --no-sync python -m benchmarks.reports.2026-08-23-daft-comparison.capability_probe

or directly::

    uv run --no-sync python benchmarks/reports/2026-08-23-daft-comparison/capability_probe.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

import numpy as np

BANNER = "=" * 78


def _png(height: int = 32, width: int = 32) -> bytes:
    """
    Build a deterministic RGB PNG for probing.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        PNG-encoded bytes.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _run_isolated(source: str) -> tuple[bool, str]:
    """
    Run a probe in a subprocess so a Rust-side panic cannot abort this script.

    Args:
        source: Python source to execute. It should print a single line.

    Returns:
        ``(survived, output)`` where ``survived`` is False if the child died.
    """
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip().splitlines()
        tail = detail[-1] if detail else "(no output)"
        return False, f"process died (exit {proc.returncode}): {tail[:160]}"
    return True, proc.stdout.strip()


def probe_versions() -> None:
    """Print the versions everything below was measured against."""
    import daft
    import polars

    import polars_cv

    print(BANNER)
    print("VERSIONS")
    print(BANNER)
    print(f"  daft       {daft.__version__}")
    print(f"  polars-cv  {polars_cv.build_info()['version']}")
    print(f"  polars     {polars.__version__}")
    print(f"  numpy      {np.__version__}")


def probe_op_surface() -> None:
    """Compare how many image operations each engine exposes natively."""
    import daft
    import daft.functions as daft_functions

    from polars_cv import Pipeline

    print(f"\n{BANNER}\nOPERATION SURFACE\n{BANNER}")

    expr = daft.col("x")
    # Daft's vision surface: image-namespace methods on Expression. As of 0.7
    # these are flat methods, not a `.image` accessor.
    daft_image_ops = sorted(
        name
        for name in dir(expr)
        if not name.startswith("_")
        and (
            "image" in name
            or name in {"resize", "crop", "encode", "decode", "convert_image"}
        )
    )
    daft_all = sorted(n for n in dir(daft_functions) if not n.startswith("_"))

    print(f"  polars-cv Pipeline ops   : {len(Pipeline.OP_NAMES)}")
    print(f"  Daft image expressions   : {len(daft_image_ops)}")
    print(f"  Daft functions (all)     : {len(daft_all)}")
    print(f"\n  Daft image ops: {', '.join(daft_image_ops)}")

    # Which of polars-cv's ops Daft has a native equivalent for.
    equivalents = {"resize", "crop", "grayscale", "perceptual_hash", "cast"}
    print(f"\n  polars-cv ops with a native Daft equivalent: {len(equivalents)}")
    print(f"    {', '.join(sorted(equivalents))}")
    missing = sorted(Pipeline.OP_NAMES - equivalents)
    print(f"  polars-cv ops with no native Daft equivalent: {len(missing)}")
    print(
        textwrap.fill(
            ", ".join(missing),
            width=74,
            initial_indent="    ",
            subsequent_indent="    ",
        )
    )


def probe_luma() -> None:
    """Recover each engine's RGB->gray weights from pure R/G/B patches."""
    import daft
    import polars as pl
    from PIL import Image

    import polars_cv.expressions  # noqa: F401
    from polars_cv import Pipeline, numpy_from_struct

    print(f"\n{BANNER}\nGRAYSCALE CONVENTION\n{BANNER}")

    def patch(rgb: tuple[int, int, int]) -> bytes:
        arr = np.zeros((4, 4, 3), np.uint8)
        arr[:, :] = rgb
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    primaries = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    daft_weights = []
    for rgb in primaries:
        frame = daft.from_pydict({"i": [patch(rgb)]}).with_column(
            "g", daft.col("i").decode_image().convert_image("L")
        )
        daft_weights.append(int(np.asarray(frame.to_pydict()["g"][0])[0, 0, 0]) / 255)

    pipe = Pipeline().source("image_bytes").grayscale()
    frame = pl.DataFrame({"i": [patch(rgb) for rgb in primaries]}).with_columns(
        g=pl.col("i").cv.pipe(pipe).sink("numpy")
    )
    pcv_weights = [
        float(numpy_from_struct(cell).reshape(-1)[0]) / 255 for cell in frame["g"]
    ]

    fmt = "  {:<24s} R={:.4f}  G={:.4f}  B={:.4f}"
    print(fmt.format("Daft convert_image('L')", *daft_weights))
    print(fmt.format("polars-cv grayscale()", *pcv_weights))
    print(fmt.format("ITU-R BT.601 (OpenCV/PIL)", 0.299, 0.587, 0.114))
    print(fmt.format("ITU-R BT.709", 0.2126, 0.7152, 0.0722))


def probe_per_row_params() -> None:
    """Check whether operation parameters may vary per row on each engine."""
    import daft
    import polars as pl

    import polars_cv.expressions  # noqa: F401
    from polars_cv import Pipeline, numpy_from_struct

    print(f"\n{BANNER}\nPER-ROW OPERATION PARAMETERS\n{BANNER}")

    encoded = [_png(64, 64), _png(64, 64)]
    frame = daft.from_pydict(
        {
            "i": encoded,
            "h": [32, 16],
            "w": [48, 24],
            "bb": [[0, 0, 20, 20], [4, 4, 10, 10]],
        }
    ).with_column("d", daft.col("i").decode_image())

    results: list[tuple[str, str, str]] = []

    try:
        out = frame.with_column(
            "r", daft.col("d").resize(daft.col("w"), daft.col("h"))
        ).to_pydict()
        shapes = [np.asarray(x).shape for x in out["r"]]
        results.append(("Daft", "resize(w, h) from columns", f"OK {shapes}"))
    except Exception as exc:  # noqa: BLE001 - reporting the failure is the point
        results.append(
            ("Daft", "resize(w, h) from columns", f"REJECTED ({type(exc).__name__})")
        )

    try:
        bbox = daft.col("bb").cast(
            daft.DataType.fixed_size_list(daft.DataType.uint32(), 4)
        )
        out = frame.with_column("c", daft.col("d").crop(bbox)).to_pydict()
        shapes = [np.asarray(x).shape for x in out["c"]]
        results.append(("Daft", "crop(bbox) from a column", f"OK {shapes}"))
    except Exception as exc:  # noqa: BLE001
        results.append(
            ("Daft", "crop(bbox) from a column", f"REJECTED ({type(exc).__name__})")
        )

    pl_frame = pl.DataFrame({"i": encoded, "h": [32, 16], "w": [48, 24]})

    pipe = (
        Pipeline().source("image_bytes").resize(height=pl.col("h"), width=pl.col("w"))
    )
    try:
        out = pl_frame.with_columns(r=pl.col("i").cv.pipe(pipe).sink("numpy"))
        shapes = [numpy_from_struct(cell).shape for cell in out["r"]]
        results.append(
            ("polars-cv", "resize(height=, width=) from columns", f"OK {shapes}")
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            ("polars-cv", "resize(height=, width=) from columns", f"REJECTED ({exc})")
        )

    pipe = (
        Pipeline()
        .source("image_bytes")
        .crop(top=0, left=0, height=pl.col("h"), width=pl.col("w"))
    )
    try:
        out = pl_frame.with_columns(c=pl.col("i").cv.pipe(pipe).sink("numpy"))
        shapes = [numpy_from_struct(cell).shape for cell in out["c"]]
        results.append(
            ("polars-cv", "crop(height=, width=) from columns", f"OK {shapes}")
        )
    except Exception as exc:  # noqa: BLE001
        results.append(
            ("polars-cv", "crop(height=, width=) from columns", f"REJECTED ({exc})")
        )

    pipe = (
        Pipeline().source("image_bytes").resize(height=32, width=32, filter=pl.col("f"))
    )
    try:
        out = pl_frame.with_columns(f=pl.Series(["bilinear", "nearest"])).with_columns(
            r=pl.col("i").cv.pipe(pipe).sink("numpy")
        )
        shapes = [numpy_from_struct(cell).shape for cell in out["r"]]
        results.append(("polars-cv", "resize(filter=) from a column", f"OK {shapes}"))
    except Exception as exc:  # noqa: BLE001
        results.append(
            ("polars-cv", "resize(filter=) from a column", f"REJECTED {str(exc)[:60]}")
        )

    for engine, what, outcome in results:
        print(f"  {engine:<10s} {what:<38s} {outcome}")


def probe_type_system() -> None:
    """
    Exercise Daft's image/tensor type system at its edges.

    Two of these abort the process rather than raising, so each runs isolated.
    """
    print(f"\n{BANNER}\nDAFT TYPE SYSTEM EDGES\n{BANNER}")

    probes: list[tuple[str, str]] = [
        (
            "arithmetic on an Image column",
            """
            import daft, numpy as np, io
            from PIL import Image
            a = np.zeros((8, 8, 3), np.uint8)
            b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
            df = daft.from_pydict({"i": [b.getvalue()]}).with_column("d", daft.col("i").decode_image())
            try:
                df.with_column("z", daft.col("d") * 2).to_pydict()
                print("OK")
            except Exception as e:
                print("REJECTED: " + str(e).strip().splitlines()[-1][:90])
            """,
        ),
        (
            "arithmetic on a Tensor column",
            """
            import daft, numpy as np
            df = daft.from_pydict({"t": [np.zeros((2, 2, 3), np.uint8)]})
            try:
                df.with_column("z", daft.col("t") * 2).to_pydict()
                print("OK")
            except Exception as e:
                print("REJECTED: " + str(e).strip().splitlines()[-1][:90])
            """,
        ),
        (
            "cast a fixed-shape tensor's dtype",
            """
            import daft, numpy as np
            dt = daft.DataType.tensor(daft.DataType.uint8(), (2, 2, 3))
            df = daft.from_pydict({"t": [np.zeros((2, 2, 3), np.uint8)]}).with_column("t", daft.col("t").cast(dt))
            target = daft.DataType.tensor(daft.DataType.float32(), (2, 2, 3))
            try:
                df.with_column("f", daft.col("t").cast(target)).to_pydict()
                print("OK")
            except Exception as e:
                print("REJECTED: " + str(e).strip().splitlines()[-1][:90])
            """,
        ),
        (
            "image_to_tensor() then cast to float32",
            """
            import daft, numpy as np, io
            from PIL import Image
            a = np.zeros((8, 8, 3), np.uint8)
            b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
            df = (daft.from_pydict({"i": [b.getvalue()]})
                  .with_column("d", daft.col("i").decode_image())
                  .with_column("t", daft.col("d").resize(4, 4).image_to_tensor()))
            target = daft.DataType.tensor(daft.DataType.float32())
            df.with_column("f", daft.col("t").cast(target)).to_pydict()
            print("OK")
            """,
        ),
        (
            "UDF returning float32 under image('RGB32F')",
            """
            import daft, numpy as np, io
            from PIL import Image
            a = np.zeros((8, 8, 3), np.uint8)
            b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
            df = daft.from_pydict({"i": [b.getvalue()]}).with_column("d", daft.col("i").decode_image())
            @daft.func.batch(return_dtype=daft.DataType.image("RGB32F"))
            def norm(s):
                return [(np.asarray(x) / 255).astype(np.float32) for x in s.to_pylist()]
            df.with_column("z", norm(daft.col("d"))).to_pydict()
            print("OK")
            """,
        ),
        (
            "UDF returning float32 under tensor(float32)",
            """
            import daft, numpy as np, io
            from PIL import Image
            a = np.zeros((8, 8, 3), np.uint8)
            b = io.BytesIO(); Image.fromarray(a).save(b, "PNG")
            df = daft.from_pydict({"i": [b.getvalue()]}).with_column("d", daft.col("i").decode_image())
            @daft.func.batch(return_dtype=daft.DataType.tensor(daft.DataType.float32()))
            def norm(s):
                return [(np.asarray(x) / 255).astype(np.float32) for x in s.to_pylist()]
            df.with_column("z", norm(daft.col("d"))).to_pydict()
            print("OK")
            """,
        ),
    ]

    for label, source in probes:
        survived, output = _run_isolated(source)
        status = output if survived else f"CRASHED — {output}"
        print(f"  {label:<44s} {status}")


def main() -> int:
    """
    Run every probe.

    Returns:
        Process exit code.
    """
    probe_versions()
    probe_op_surface()
    probe_luma()
    probe_per_row_params()
    probe_type_system()
    print(f"\n{BANNER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
