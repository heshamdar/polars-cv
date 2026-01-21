# Pipeline

The `Pipeline` class is the core builder for image and array processing pipelines.

## Overview

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
)
```

## API Reference

::: polars_cv.Pipeline
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

