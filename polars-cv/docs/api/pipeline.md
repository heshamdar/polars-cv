# Pipeline

The `Pipeline` class is the core builder for image/array processing pipelines.

## Overview

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
    .sink("png")
)
```

## API Reference

::: polars_cv.Pipeline
    options:
      members:
        - source
        - sink
        - resize
        - grayscale
        - blur
        - threshold
        - crop
        - flip_h
        - flip_v
        - normalize
        - scale
        - clamp
        - relu
        - cast
        - assert_shape
        - perceptual_hash
      show_root_heading: true
      show_source: false
      heading_level: 3

