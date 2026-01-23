import polars as pl
import pytest
from polars_cv import Pipeline

def test_lazy_schema_resize_list():
    """Test that resize correctly updates schema for list sink."""
    pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
    
    df = pl.DataFrame({"image": [b""]})
    # Use a dummy column to avoid issues with empty binary if needed
    # But collect_schema should really just work.
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("list")).collect_schema()
    
    # Expected: List(List(List(UInt8))) for 100x200x3
    expected_type = pl.List(pl.List(pl.List(pl.UInt8)))
    assert schema["image"] == expected_type

def test_lazy_schema_resize_array():
    """Test that resize correctly updates schema for array sink."""
    pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
    
    df = pl.DataFrame({"image": [b""]})
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("array", shape=[100, 200, 3])).collect_schema()
    
    # Expected: Array(Array(Array(UInt8, 3), 200), 100)
    expected_type = pl.Array(pl.Array(pl.Array(pl.UInt8, 3), 200), 100)
    assert schema["image"] == expected_type

def test_lazy_schema_cast():
    """Test that cast correctly updates schema."""
    pipe = Pipeline().source("image_bytes").resize(height=100, width=200).cast("f32")
    
    df = pl.DataFrame({"image": [b""]})
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("list")).collect_schema()
    
    # Expected: List(List(List(Float32)))
    expected_type = pl.List(pl.List(pl.List(pl.Float32)))
    assert schema["image"] == expected_type

def test_lazy_schema_grayscale():
    """Test that grayscale updates channels in schema."""
    pipe = Pipeline().source("image_bytes").resize(height=100, width=200).grayscale()
    
    df = pl.DataFrame({"image": [b""]})
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("list")).collect_schema()
    
    # Expected: List(List(List(UInt8))) where inner dimension is 1
    expected_type = pl.List(pl.List(pl.List(pl.UInt8)))
    assert schema["image"] == expected_type

def test_lazy_schema_assert_shape():
    """Test that assert_shape provides schema info."""
    pipe = Pipeline().source("image_bytes").assert_shape(height=128, width=128, channels=3)
    
    df = pl.DataFrame({"image": [b""]})
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("array", shape=[128, 128, 3])).collect_schema()
    
    expected_type = pl.Array(pl.Array(pl.Array(pl.UInt8, 3), 128), 128)
    assert schema["image"] == expected_type

def test_lazy_schema_complex_chain():
    """Test a complex chain of operations."""
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=224, width=224)
        .grayscale()
        .cast("f64")
    )
    
    df = pl.DataFrame({"image": [b""]})
    schema = df.lazy().select(pl.col("image").cv.pipe(pipe).sink("list")).collect_schema()
    
    expected_type = pl.List(pl.List(pl.List(pl.Float64)))
    assert schema["image"] == expected_type
