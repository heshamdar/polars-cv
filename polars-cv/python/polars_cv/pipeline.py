"""
Pipeline builder for polars-cv.

This module provides the Pipeline class for building lazy image/array
processing pipelines that can be applied to Polars DataFrame columns.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

from polars_cv._types import (
    OPERATION_OUTPUT_DTYPE,
    CloudOptions,
    DType,
    FilterType,
    FloatOrExpr,
    HashAlgorithm,
    HistogramOutput,
    IntOrExpr,
    NormalizeMethod,
    OpSpec,
    OutputDType,
    PadMode,
    PadPosition,
    ParamValue,
    ShapeHints,
    SinkFormat,
    SinkSpec,
    SourceFormat,
    SourceSpec,
)

if TYPE_CHECKING:
    from polars_cv._graph import PipelineGraph
    from polars_cv.lazy import LazyPipelineExpr


class Pipeline:
    """
    Modular pipeline builder for image and array operations.

    A pipeline defines a sequence of operations that can be applied to a Polars
    expression using the `.cv.pipe()` accessor. The pipeline is executed when
    `.sink()` is called on the resulting expression.

    All operations accept either literal values or Polars expressions.
    Expressions are resolved at execution time per row.

    Example:
        ```python
        >>> from polars_cv import Pipeline
        >>> import polars as pl
        >>>
        >>> # Define a reusable pipeline (without a sink)
        >>> preprocess = (
        ...     Pipeline()
        ...     .source("image_bytes")
        ...     .resize(height=224, width=224)
        ...     .grayscale()
        ... )
        >>>
        >>> # Apply to a DataFrame and choose the output format at the sink
        >>> df = pl.DataFrame({"image": [img_bytes]})
        >>> result = df.with_columns(
        ...     processed=pl.col("image").cv.pipe(preprocess).sink("numpy")
        ... )
        ```

    Pipelines support typed domain tracking for transitions between images,
    geometry, and numeric results:
    - buffer: Image/array data (default)
    - contour: Polygon geometry
    - scalar: Single numeric values
    - vector: Multiple numeric values (e.g., bounding boxes)
    """

    # Domain constants
    DOMAIN_BUFFER = "buffer"
    DOMAIN_CONTOUR = "contour"
    DOMAIN_SCALAR = "scalar"
    DOMAIN_VECTOR = "vector"

    # Mapping of operations to the domain they produce
    # Operations not listed here preserve the current domain
    # Note: reduce_argmax/reduce_argmin always preserve buffer domain (reduced shape)
    _OPERATION_OUTPUT_DOMAIN: dict[str, str] = {
        "extract_shape": DOMAIN_VECTOR,
        "perceptual_hash": DOMAIN_VECTOR,
        "histogram": DOMAIN_VECTOR,  # For most modes, quantized preserves buffer
        "reduce_sum": DOMAIN_SCALAR,
        "reduce_max": DOMAIN_SCALAR,  # When axis=None
        "reduce_min": DOMAIN_SCALAR,
        "reduce_mean": DOMAIN_SCALAR,
        "reduce_std": DOMAIN_SCALAR,
        "reduce_popcount": DOMAIN_SCALAR,
        # reduce_argmax/reduce_argmin are NOT here - they always preserve buffer domain
        "extract_contours": DOMAIN_CONTOUR,
        "rasterize": DOMAIN_BUFFER,
        "contour_area": DOMAIN_SCALAR,
        "contour_perimeter": DOMAIN_SCALAR,
        "contour_centroid": DOMAIN_VECTOR,
        "contour_bounding_box": DOMAIN_VECTOR,
    }

    def __init__(self) -> None:
        """Initialize an empty pipeline."""
        self._source: SourceSpec | None = None
        self._shape_hints: ShapeHints = ShapeHints()
        self._ops: list[OpSpec] = []
        self._sink: SinkSpec | None = None
        self._expr_refs: list[pl.Expr] = []
        # Domain tracking for typed pipelines
        self._current_domain: str = self.DOMAIN_BUFFER
        # Output dtype tracking - starts as "u8" for image sources
        self._output_dtype: str = "u8"
        # Number of dimensions tracking
        self._expected_ndim: int | None = None
        # Whether dtype/ndim should be auto-inferred from the Polars column at planning time
        self._auto_infer_from_input: bool = False

    @staticmethod
    def _compute_output_domain_dtype_ndim(
        ops: list["OpSpec"],
        initial_domain: str = "buffer",
        initial_dtype: str = "u8",
        initial_ndim: int | None = None,
    ) -> tuple[str, str, int | None]:
        """
        Compute the output domain, dtype, and ndim after applying operations.

        This is used for static type inference to ensure planning-time types
        match runtime types.

        Args:
            ops: Sequence of operations to analyze.
            initial_domain: Starting domain (default: buffer for image sources).
            initial_dtype: Starting dtype (default: u8 for image sources).
            initial_ndim: Starting number of dimensions.

        Returns:
            Tuple of (output_domain, output_dtype, output_ndim) after all operations.
        """
        domain = initial_domain
        dtype = initial_dtype
        ndim = initial_ndim

        for op_spec in ops:
            op_name = op_spec.op

            # Update domain if this operation changes it
            if op_name in Pipeline._OPERATION_OUTPUT_DOMAIN:
                domain = Pipeline._OPERATION_OUTPUT_DOMAIN[op_name]

            # Update dtype if this operation is in the dtype mapping
            if op_name in OPERATION_OUTPUT_DTYPE:
                dtype = OPERATION_OUTPUT_DTYPE[op_name]

            # Handle special cases for cast (param-dependent)
            if op_name == "cast":
                dtype_param = op_spec.params.get("dtype")
                if dtype_param and not dtype_param.is_expr:
                    dtype = dtype_param.value

            # Handle special cases for histogram (mode-dependent)
            if op_name == "histogram":
                # Check for output_mode param
                mode_param = op_spec.params.get("output_mode")
                if mode_param and not mode_param.is_expr:
                    mode = mode_param.value
                    # Quantized mode keeps buffer domain
                    if mode == 3:  # QUANTIZED enum value
                        domain = Pipeline.DOMAIN_BUFFER
                        dtype = "u32"
                        # ndim remains same
                    elif mode == 0:  # COUNTS
                        dtype = "u64"
                        ndim = 1
                    else:  # NORMALIZED or EDGES
                        dtype = "f64"
                        ndim = 1
                else:
                    # Default mode is counts -> ndim=1
                    ndim = 1

            # Handle axis-based reductions that keep buffer domain
            if op_name in ("reduce_max", "reduce_min", "reduce_mean", "reduce_std"):
                axis_param = op_spec.params.get("axis")
                if (
                    axis_param
                    and not axis_param.is_expr
                    and axis_param.value is not None
                ):
                    # Axis reduction keeps buffer domain but reduces ndim
                    domain = Pipeline.DOMAIN_BUFFER
                    if ndim is not None:
                        ndim = max(0, ndim - 1)
                else:
                    # Global reduction -> scalar
                    domain = Pipeline.DOMAIN_SCALAR
                    ndim = 0

            # Handle global reductions
            if op_name in ("reduce_sum", "reduce_popcount"):
                domain = Pipeline.DOMAIN_SCALAR
                ndim = 0

            # Handle other domain changes
            if domain == Pipeline.DOMAIN_SCALAR:
                ndim = 0
            elif domain == Pipeline.DOMAIN_VECTOR:
                ndim = 1
            elif op_name == "perceptual_hash":
                ndim = 1
            elif op_name == "extract_shape":
                ndim = 1
            elif op_name == "rasterize":
                ndim = 3

        return domain, dtype, ndim

    def _track_expr(self, value: IntOrExpr | FloatOrExpr) -> ParamValue:
        """
        Create a ParamValue and track the expression if needed.

        Args:
            value: Literal or expression value.

        Returns:
            ParamValue instance.
        """
        param = ParamValue.from_arg(value)
        if param.is_expr and isinstance(value, pl.Expr):
            # Check if we already track this expression
            expr_str = str(value)
            if not any(str(e) == expr_str for e in self._expr_refs):
                self._expr_refs.append(value)
        return param

    def _clone(self) -> "Pipeline":
        """Create a shallow clone of this pipeline for chaining."""
        new = Pipeline()
        new._source = self._source
        new._shape_hints = self._shape_hints
        new._ops = self._ops.copy()
        new._sink = self._sink
        new._expr_refs = self._expr_refs.copy()
        new._current_domain = self._current_domain
        new._output_dtype = self._output_dtype
        new._expected_ndim = self._expected_ndim
        new._auto_infer_from_input = self._auto_infer_from_input
        return new

    def _source_equal(self, other: "Pipeline") -> bool:
        """
        Check if two pipelines have equivalent sources.

        Used by CSE optimization to determine if pipelines can share
        a common prefix.

        Args:
            other: Another Pipeline to compare with.

        Returns:
            True if both pipelines have the same source specification.
        """
        if self._source is None or other._source is None:
            return self._source is None and other._source is None
        return self._source == other._source

    def _validate_domain(self, expected: str, op_name: str) -> None:
        """
        Validate that the current domain matches the expected domain.

        Args:
            expected: Expected domain ("buffer", "contour", "scalar", "vector").
            op_name: Name of the operation for error messages.

        Raises:
            ValueError: If current domain doesn't match expected.
        """
        if self._current_domain != expected:
            raise ValueError(
                f"{op_name}() expects {expected} input but pipeline is currently in "
                f"{self._current_domain} domain. Add a domain-converting operation "
                f"(e.g., rasterize() for contour→buffer, extract_contours() for buffer→contour)."
            )

    def current_domain(self) -> str:
        """
        Get the current data domain of the pipeline.

        Returns:
            Current domain: "buffer", "contour", "scalar", or "vector".
        """
        return self._current_domain

    def output_dtype(self) -> str:
        """
        Get the expected output dtype of the pipeline.

        This is the dtype of the buffer after all operations have been applied.
        Used for static type inference in list/array sinks.

        Returns:
            Output dtype string: "u8", "f32", "f64", etc.
        """
        return self._output_dtype

    def _update_output_dtype(self, op_name: str) -> None:
        """
        Update the output dtype based on the operation being added.

        Args:
            op_name: Name of the operation being added.
        """
        # Re-compute from all operations to handle parameter-dependent dtypes like cast
        _, self._output_dtype, self._expected_ndim = (
            self._compute_output_domain_dtype_ndim(
                self._ops,
                initial_domain=self._current_domain,
                initial_dtype=self._output_dtype,
                initial_ndim=self._expected_ndim,
            )
        )

    def _update_shape_hints(self, op_name: str, params: dict[str, ParamValue]) -> None:
        """
        Update shape hints based on the operation being added.

        Args:
            op_name: Name of the operation.
            params: Parameters of the operation.
        """
        if op_name == "resize":
            h = params.get("height")
            w = params.get("width")
            if h and not h.is_expr:
                self._shape_hints.height = h
            if w and not w.is_expr:
                self._shape_hints.width = w
        elif op_name == "grayscale":
            self._shape_hints.channels = ParamValue(is_expr=False, value=1)
        elif op_name == "pad":
            # If we have current hints and literal padding, we can update
            if (
                self._shape_hints.height
                and not self._shape_hints.height.is_expr
                and self._shape_hints.width
                and not self._shape_hints.width.is_expr
            ):
                top = params.get("top")
                bottom = params.get("bottom")
                left = params.get("left")
                right = params.get("right")

                if (
                    top
                    and not top.is_expr
                    and bottom
                    and not bottom.is_expr
                    and left
                    and not left.is_expr
                    and right
                    and not right.is_expr
                ):
                    self._shape_hints.height = ParamValue(
                        is_expr=False,
                        value=self._shape_hints.height.value + top.value + bottom.value,
                    )
                    self._shape_hints.width = ParamValue(
                        is_expr=False,
                        value=self._shape_hints.width.value + left.value + right.value,
                    )
        elif op_name == "pad_to_size" or op_name == "letterbox":
            h = params.get("height")
            w = params.get("width")
            if h and not h.is_expr:
                self._shape_hints.height = h
            if w and not w.is_expr:
                self._shape_hints.width = w
        elif op_name == "crop":
            h = params.get("height")
            w = params.get("width")
            if h and not h.is_expr:
                self._shape_hints.height = h
            if w and not w.is_expr:
                self._shape_hints.width = w
        elif op_name == "reshape":
            # shape param is a ParamValue(is_expr=False, value=[dict, dict, ...])
            shape_val = params.get("shape")
            if shape_val and not shape_val.is_expr:
                shape_list = shape_val.value
                if len(shape_list) >= 2:
                    h_dict = shape_list[0]
                    w_dict = shape_list[1]
                    if h_dict["type"] == "literal":
                        self._shape_hints.height = ParamValue(
                            is_expr=False, value=h_dict["value"]
                        )
                    if w_dict["type"] == "literal":
                        self._shape_hints.width = ParamValue(
                            is_expr=False, value=w_dict["value"]
                        )
                    if len(shape_list) >= 3:
                        c_dict = shape_list[2]
                        if c_dict["type"] == "literal":
                            self._shape_hints.channels = ParamValue(
                                is_expr=False, value=c_dict["value"]
                            )
        elif op_name == "rotate":
            angle = params.get("angle")
            expand = params.get("expand")
            if (
                angle
                and not angle.is_expr
                and expand
                and not expand.is_expr
                and not expand.value
            ):
                # Non-expanding rotation
                if angle.value in (90, 270, -90, -270):
                    # Swap height and width
                    h = self._shape_hints.height
                    w = self._shape_hints.width
                    self._shape_hints.height = w
                    self._shape_hints.width = h

    # --- Source (required, starts the chain) ---

    def source(
        self,
        format: str = "image_bytes",
        *,
        dtype: str | None = None,
        # Contour source parameters
        width: IntOrExpr | None = None,
        height: IntOrExpr | None = None,
        shape: "LazyPipelineExpr | None" = None,
        fill_value: int = 255,
        background: int = 0,
        # Cloud storage options for file_path sources
        cloud_options: "CloudOptions | dict[str, Any] | None" = None,
        # Contiguity option for list/array sources
        require_contiguous: bool = False,
    ) -> "Pipeline":
        """
        Define the input source format.

        Args:
            format: How to interpret input data.
                - "image_bytes": Decode PNG/JPEG (auto-detect)
                - "blob": VIEW protocol binary (self-describing)
                - "raw": Raw bytes (requires dtype)
                - "list": Polars nested List column
                - "array": Polars fixed-size Array column
                - "file_path": Read from path (local, s3://, gs://, az://, http://)
                - "contour": Rasterize contour struct to binary mask
            dtype: Data type for "raw" format (required), or override for "list"/"array".
            width: Output mask width for "contour" format.
            height: Output mask height for "contour" format.
            shape: Infer dimensions from another pipeline for "contour" format.
            fill_value: Value for pixels inside contour (default 255).
            background: Value for pixels outside contour (default 0).
            cloud_options: Credentials for cloud storage (S3, GCS, Azure).
            require_contiguous: For "list"/"array", whether to require rectangular data.

        Example:
            ```python
            >>> # Decode PNG/JPEG bytes from a column
            >>> pipe = Pipeline().source("image_bytes").resize(224, 224)
            >>>
            >>> # Read from file paths or URLs
            >>> df = pl.DataFrame({"url": ["https://example.com/image.png"]})
            >>> pipe = Pipeline().source("file_path").grayscale()
            >>> expr = pl.col("url").cv.pipe(pipe).sink("numpy")
            ```
        """
        from polars_cv.lazy import LazyPipelineExpr

        new = self._clone()
        try:
            fmt = SourceFormat(format)
        except ValueError as e:
            valid = [f.value for f in SourceFormat]
            msg = f"Invalid source format '{format}'. Valid: {valid}"
            raise ValueError(msg) from e

        dtype_enum = None
        if dtype is not None:
            try:
                dtype_enum = DType(dtype)
            except ValueError as e:
                valid = [d.value for d in DType]
                msg = f"Invalid dtype '{dtype}'. Valid: {valid}"
                raise ValueError(msg) from e

        # RAW format always requires dtype (no type metadata in raw bytes)
        # LIST and ARRAY can auto-infer dtype from Polars column type
        if fmt == SourceFormat.RAW and dtype_enum is None:
            msg = "dtype is required for 'raw' source format (raw bytes have no type metadata)"
            raise ValueError(msg)

        # Handle contour source format
        if fmt == SourceFormat.CONTOUR:
            new._expected_ndim = 3  # Rasterized mask is 3D (H, W, 1)
            has_explicit_dims = width is not None or height is not None
            has_shape = shape is not None

            if has_explicit_dims and has_shape:
                msg = (
                    "Cannot specify both 'shape' and explicit dimensions (width/height)"
                )
                raise ValueError(msg)

            if not has_explicit_dims and not has_shape:
                msg = (
                    "Contour source requires either:\n"
                    "  1. Both 'width' and 'height' parameters, or\n"
                    "  2. A 'shape' LazyPipelineExpr to infer dimensions from"
                )
                raise ValueError(msg)

            if has_explicit_dims and (width is None or height is None):
                msg = "Both 'width' and 'height' must be specified together"
                raise ValueError(msg)

            # Track expressions for width/height if they are expressions
            width_param = new._track_expr(width) if width is not None else None
            height_param = new._track_expr(height) if height is not None else None

            # Serialize shape pipeline if provided
            shape_pipeline_dict = None
            if shape is not None:
                if not isinstance(shape, LazyPipelineExpr):
                    msg = "'shape' must be a LazyPipelineExpr"
                    raise TypeError(msg)
                # Collect the graph from the shape expression
                shape_pipeline_dict = {
                    "node_id": shape._node_id,
                    "column": str(shape._column),
                    "pipeline": shape._pipeline._to_spec_dict(),
                    "upstream": [u._node_id for u in shape._upstream],
                }

            new._source = SourceSpec(
                format=fmt,
                dtype=dtype_enum,
                width=width_param,
                height=height_param,
                fill_value=fill_value,
                background=background,
                shape_pipeline=shape_pipeline_dict,
            )
        else:
            # Handle cloud_options for file_path format
            cloud_opts = None
            if fmt == SourceFormat.FILE_PATH and cloud_options is not None:
                if isinstance(cloud_options, CloudOptions):
                    cloud_opts = cloud_options
                elif isinstance(cloud_options, dict):
                    # Convert dict to CloudOptions, handling type conversions
                    opts_dict = dict(cloud_options)
                    # Convert "anonymous" from string if present
                    if "anonymous" in opts_dict and isinstance(
                        opts_dict["anonymous"], str
                    ):
                        opts_dict["anonymous"] = (
                            opts_dict["anonymous"].lower() == "true"
                        )
                    cloud_opts = CloudOptions(**opts_dict)
                else:
                    msg = (
                        f"cloud_options must be CloudOptions or dict, "
                        f"got {type(cloud_options)}"
                    )
                    raise TypeError(msg)

            new._source = SourceSpec(
                format=fmt,
                dtype=dtype_enum,
                cloud_options=cloud_opts,
                require_contiguous=require_contiguous,
            )
            # Default ndim for image/buffer sources
            if fmt in (
                SourceFormat.IMAGE_BYTES,
                SourceFormat.FILE_PATH,
                SourceFormat.BLOB,
                SourceFormat.RAW,
            ):
                new._expected_ndim = 3
            elif fmt in (SourceFormat.LIST, SourceFormat.ARRAY):
                # For list/array sources, infer dtype and ndim from the
                # Polars column at planning time when not explicitly given.
                if dtype_enum is not None:
                    # User provided explicit dtype — use it, default ndim=3
                    new._expected_ndim = 3
                else:
                    # Mark as "auto" so Rust resolves from input_fields
                    new._output_dtype = "auto"
                    new._expected_ndim = None
                    new._auto_infer_from_input = True

        return new

    # --- Shape Assertions (optional, helps planner) ---

    def assert_shape(
        self,
        *,
        height: IntOrExpr | None = None,
        width: IntOrExpr | None = None,
        channels: IntOrExpr | None = None,
        batch: IntOrExpr | None = None,
    ) -> "Pipeline":
        """
        Provide shape hints for the pipeline.

        Expressions are resolved per-row at execution time.
        Literal values help the planner optimize.

        Args:
            height: Image height (literal or expression).
            width: Image width (literal or expression).
            channels: Number of channels (literal or expression).
            batch: Batch size (literal or expression).

        Returns:
            Self for chaining.
        """
        new = self._clone()
        if height is not None:
            new._shape_hints.height = new._track_expr(height)
        if width is not None:
            new._shape_hints.width = new._track_expr(width)
        if channels is not None:
            new._shape_hints.channels = new._track_expr(channels)
        if batch is not None:
            new._shape_hints.batch = new._track_expr(batch)
        return new

    # --- View Operations (zero-copy where possible) ---

    def transpose(self, axes: list[int]) -> "Pipeline":
        """
        Transpose dimensions.

        Args:
            axes: New order of axes.

        Returns:
            Self for chaining.
        """
        new = self._clone()
        # Axes are always literals (list of ints)
        new._ops.append(
            OpSpec(
                op="transpose",
                params={"axes": ParamValue(is_expr=False, value=axes)},
            )
        )
        return new

    def reshape(self, shape: list[int | pl.Expr]) -> "Pipeline":
        """
        Reshape array to new dimensions.

        Args:
            shape: New shape (list of ints or expressions).

        Returns:
            Self for chaining.
        """
        new = self._clone()
        # Handle mixed literal/expr shapes
        shape_params = [new._track_expr(s) for s in shape]
        new._ops.append(
            OpSpec(
                op="reshape",
                params={
                    "shape": ParamValue(
                        is_expr=False,
                        value=[p.to_dict() for p in shape_params],
                    )
                },
            )
        )
        new._update_shape_hints("reshape", new._ops[-1].params)
        return new

    def flip(self, axes: list[int]) -> "Pipeline":
        """
        Flip along specified axes.

        Args:
            axes: Axes to flip.

        Returns:
            Self for chaining.
        """
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="flip",
                params={"axes": ParamValue(is_expr=False, value=axes)},
            )
        )
        return new

    def flip_h(self) -> "Pipeline":
        """
        Flip horizontally (along width axis).

        Returns:
            Self for chaining.
        """
        return self.flip(axes=[1])

    def flip_v(self) -> "Pipeline":
        """
        Flip vertically (along height axis).

        Returns:
            Self for chaining.
        """
        return self.flip(axes=[0])

    def crop(
        self,
        *,
        top: IntOrExpr = 0,
        left: IntOrExpr = 0,
        height: IntOrExpr | None = None,
        width: IntOrExpr | None = None,
    ) -> "Pipeline":
        """
        Extract a rectangular region.

        Args:
            top: Top offset.
            left: Left offset.
            height: Crop height (None = to end).
            width: Crop width (None = to end).
        """
        new = self._clone()
        params: dict[str, ParamValue] = {
            "top": new._track_expr(top),
            "left": new._track_expr(left),
        }
        if height is not None:
            params["height"] = new._track_expr(height)
        if width is not None:
            params["width"] = new._track_expr(width)

        new._ops.append(OpSpec(op="crop", params=params))
        new._update_shape_hints("crop", new._ops[-1].params)
        return new

    # --- Compute Operations ---

    def cast(self, dtype: str) -> "Pipeline":
        """
        Cast to a different data type.

        Args:
            dtype: Target data type (e.g., "f32", "u8").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If dtype is invalid.
        """
        new = self._clone()
        try:
            dtype_enum = DType(dtype)
        except ValueError as e:
            valid = [d.value for d in DType]
            msg = f"Invalid dtype '{dtype}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="cast",
                params={"dtype": ParamValue(is_expr=False, value=dtype_enum.value)},
            )
        )
        new._update_output_dtype("cast")
        return new

    def scale(
        self,
        factor: FloatOrExpr,
        out_dtype: str | None = None,
    ) -> "Pipeline":
        """
        Multiply all values by a factor.

        Args:
            factor: Scale factor.
            out_dtype: Output type (promotes to f32 if None and input is int).
        """
        new = self._clone()
        params: dict[str, ParamValue] = {
            "factor": new._track_expr(factor),
        }

        # Add out_dtype if specified
        if out_dtype is not None:
            try:
                out_dtype_enum = OutputDType(out_dtype)
            except ValueError as e:
                valid = [d.value for d in OutputDType]
                msg = f"Invalid out_dtype '{out_dtype}'. Valid: {valid}"
                raise ValueError(msg) from e
            params["out_dtype"] = ParamValue(is_expr=False, value=out_dtype_enum.value)

        new._ops.append(OpSpec(op="scale", params=params))
        new._update_output_dtype("scale")
        return new

    def normalize(
        self,
        method: str = "minmax",
        mean: list[float] | None = None,
        std: list[float] | None = None,
        out_dtype: str | None = None,
    ) -> "Pipeline":
        """
        Normalize values to a standard range.

        Args:
            method: "minmax" (scale to [0,1]) or "zscore" (mean=0, std=1).
            out_dtype: Output type (default "f32").

        Example:
            >>> Pipeline().source().normalize(method="minmax")
        """
        new = self._clone()
        try:
            method_enum = NormalizeMethod(method)
        except ValueError as e:
            valid = [m.value for m in NormalizeMethod]
            msg = f"Invalid normalize method '{method}'. Valid: {valid}"
            raise ValueError(msg) from e

        params: dict[str, ParamValue] = {
            "method": ParamValue(is_expr=False, value=method_enum.value),
        }

        # Handle preset method with mean/std
        if method_enum == NormalizeMethod.PRESET:
            if mean is None or std is None:
                msg = "method='preset' requires both 'mean' and 'std' parameters"
                raise ValueError(msg)
            if len(mean) != len(std):
                msg = f"mean length ({len(mean)}) must match std length ({len(std)})"
                raise ValueError(msg)
            params["mean"] = ParamValue(is_expr=False, value=mean)
            params["std"] = ParamValue(is_expr=False, value=std)
        elif mean is not None or std is not None:
            msg = "mean/std parameters are only valid for method='preset'"
            raise ValueError(msg)

        # Add out_dtype if specified
        if out_dtype is not None:
            try:
                out_dtype_enum = OutputDType(out_dtype)
            except ValueError as e:
                valid = [d.value for d in OutputDType]
                msg = f"Invalid out_dtype '{out_dtype}'. Valid: {valid}"
                raise ValueError(msg) from e
            params["out_dtype"] = ParamValue(is_expr=False, value=out_dtype_enum.value)

        new._ops.append(OpSpec(op="normalize", params=params))
        new._update_output_dtype("normalize")
        return new

    def clamp(
        self,
        min_val: FloatOrExpr,
        max_val: FloatOrExpr,
        out_dtype: str | None = None,
    ) -> "Pipeline":
        """
        Clamp values to a range.

        This operation accepts any numeric input dtype and automatically handles
        type promotion. Integers are promoted to float32; floats are preserved.

        Args:
            min_val: Minimum value (literal or expression).
            max_val: Maximum value (literal or expression).
            out_dtype: Output dtype. Options:
                - None: Promote integers to f32, preserve floats
                - "f32": Output float32
                - "f64": Output float64
                - "preserve": Keep input dtype (floats preserved, integers -> f32)

        Returns:
            Self for chaining.
        """
        new = self._clone()
        params: dict[str, ParamValue] = {
            "min": new._track_expr(min_val),
            "max": new._track_expr(max_val),
        }

        # Add out_dtype if specified
        if out_dtype is not None:
            try:
                out_dtype_enum = OutputDType(out_dtype)
            except ValueError as e:
                valid = [d.value for d in OutputDType]
                msg = f"Invalid out_dtype '{out_dtype}'. Valid: {valid}"
                raise ValueError(msg) from e
            params["out_dtype"] = ParamValue(is_expr=False, value=out_dtype_enum.value)

        new._ops.append(OpSpec(op="clamp", params=params))
        new._update_output_dtype("clamp")
        return new

    def relu(self) -> "Pipeline":
        """
        Apply ReLU activation (max(0, x)).

        All negative values are set to zero, positive values are unchanged.
        Works on any numeric dtype.

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").relu().sink("numpy")
            ```
        """
        new = self._clone()
        new._ops.append(OpSpec(op="relu", params={}))
        new._update_output_dtype("relu")
        return new

    # --- Image Operations ---

    def resize(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to specified dimensions.

        Args:
            height: Target height.
            width: Target width.
            filter: Interpolation: "nearest", "bilinear", "lanczos3" (default).

        Example:
            >>> Pipeline().source("image_bytes").resize(height=224, width=224)
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize")
        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize",
                params={
                    "height": new._track_expr(height),
                    "width": new._track_expr(width),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        new._update_shape_hints("resize", new._ops[-1].params)
        return new

    def resize_scale(
        self,
        *,
        scale: FloatOrExpr | None = None,
        scale_x: FloatOrExpr | None = None,
        scale_y: FloatOrExpr | None = None,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image by scale factor.

        Target dimensions are computed at runtime as:
        - new_width = input_width * scale_x
        - new_height = input_height * scale_y

        Domain: buffer → buffer

        Args:
            scale: Uniform scale factor (applies to both x and y).
            scale_x: X (width) scale factor. If None, uses scale.
            scale_y: Y (height) scale factor. If None, uses scale.
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If neither scale nor scale_x/scale_y specified.
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Uniform 50% downscale
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale=0.5)
            >>>
            >>> # Non-uniform: half width, double height
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale_x=0.5, scale_y=2.0)
            >>>
            >>> # Dynamic scale from column
            >>> pipe = Pipeline().source("image_bytes").resize_scale(scale=pl.col("zoom"))
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize_scale")

        # Resolve scale factors
        if scale is None and scale_x is None and scale_y is None:
            msg = "Must specify 'scale' or 'scale_x'/'scale_y'"
            raise ValueError(msg)

        actual_scale_x = scale_x if scale_x is not None else scale
        actual_scale_y = scale_y if scale_y is not None else scale

        if actual_scale_x is None or actual_scale_y is None:
            msg = "Must specify both scale factors or use 'scale' for uniform scaling"
            raise ValueError(msg)

        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize_scale",
                params={
                    "scale_x": new._track_expr(actual_scale_x),
                    "scale_y": new._track_expr(actual_scale_y),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        return new

    def resize_to_height(
        self,
        height: IntOrExpr,
        *,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to target height, preserving aspect ratio.

        Width is computed at runtime as: new_width = height * (input_width / input_height)

        Domain: buffer → buffer

        Args:
            height: Target height (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").resize_to_height(224)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize_to_height")
        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize_to_height",
                params={
                    "height": new._track_expr(height),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        return new

    def resize_to_width(
        self,
        width: IntOrExpr,
        *,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image to target width, preserving aspect ratio.

        Height is computed at runtime as: new_height = width * (input_height / input_width)

        Domain: buffer → buffer

        Args:
            width: Target width (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").resize_to_width(224)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize_to_width")
        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize_to_width",
                params={
                    "width": new._track_expr(width),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        return new

    def resize_max(
        self,
        max_size: IntOrExpr,
        *,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image so the maximum dimension equals target, preserving aspect ratio.

        If input is 200x100 and max_size=50, output is 50x25 (width was max, now 50).

        Domain: buffer → buffer

        Args:
            max_size: Target for the maximum dimension (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Ensure no dimension exceeds 224
            >>> pipe = Pipeline().source("image_bytes").resize_max(224)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize_max")
        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize_max",
                params={
                    "max_size": new._track_expr(max_size),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        return new

    def resize_min(
        self,
        min_size: IntOrExpr,
        *,
        filter: str = "lanczos3",
    ) -> "Pipeline":
        """
        Resize image so the minimum dimension equals target, preserving aspect ratio.

        If input is 200x100 and min_size=50, output is 100x50 (height was min, now 50).

        Domain: buffer → buffer

        Args:
            min_size: Target for the minimum dimension (literal or expression).
            filter: Resize filter ("nearest", "bilinear", "lanczos3").

        Returns:
            Self for chaining.

        Raises:
            ValueError: If filter is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Ensure min dimension is at least 224
            >>> pipe = Pipeline().source("image_bytes").resize_min(224)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "resize_min")
        new = self._clone()
        try:
            filter_enum = FilterType(filter)
        except ValueError as e:
            valid = [f.value for f in FilterType]
            msg = f"Invalid filter '{filter}'. Valid: {valid}"
            raise ValueError(msg) from e

        new._ops.append(
            OpSpec(
                op="resize_min",
                params={
                    "min_size": new._track_expr(min_size),
                    "filter": ParamValue(is_expr=False, value=filter_enum.value),
                },
            )
        )
        new._update_output_dtype("resize")
        return new

    # --- Padding Operations ---

    def pad(
        self,
        *,
        top: IntOrExpr = 0,
        bottom: IntOrExpr = 0,
        left: IntOrExpr = 0,
        right: IntOrExpr = 0,
        value: float = 0.0,
        mode: str = "constant",
    ) -> "Pipeline":
        """
        Add padding to the image.

        Domain: buffer → buffer

        Args:
            top: Padding on top edge.
            bottom: Padding on bottom edge.
            left: Padding on left edge.
            right: Padding on right edge.
            value: Fill value for "constant" mode (default 0).
            mode: Padding mode - "constant", "edge", "reflect", "symmetric".

        Returns:
            Self for chaining.

        Raises:
            ValueError: If mode is invalid or current domain is not buffer.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").pad(top=10, bottom=10)
            >>> pipe = Pipeline().source("image_bytes").pad(left=20, right=20, value=128)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "pad")

        try:
            mode_enum = PadMode(mode)
        except ValueError as e:
            valid = [m.value for m in PadMode]
            msg = f"Invalid pad mode '{mode}'. Valid: {valid}"
            raise ValueError(msg) from e

        new = self._clone()
        new._ops.append(
            OpSpec(
                op="pad",
                params={
                    "top": new._track_expr(top),
                    "bottom": new._track_expr(bottom),
                    "left": new._track_expr(left),
                    "right": new._track_expr(right),
                    "value": ParamValue(is_expr=False, value=value),
                    "mode": ParamValue(is_expr=False, value=mode_enum.value),
                },
            )
        )
        new._update_shape_hints("pad", new._ops[-1].params)
        return new

    def pad_to_size(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        position: str = "center",
        value: float = 0.0,
    ) -> "Pipeline":
        """
        Pad image to exact target size.

        Dimensions are computed at runtime. If image is larger than target,
        it will NOT be cropped - use resize first if needed.

        Domain: buffer → buffer

        Args:
            height: Target height.
            width: Target width.
            position: Where to place original content:
                - "center": Center content in padded area (default)
                - "top-left": Place at top-left corner
                - "bottom-right": Place at bottom-right corner
            value: Fill value for padding (default 0).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If position is invalid or current domain is not buffer.

        Example:
            ```python
            >>> # Pad 50x100 image to 100x200, centered
            >>> pipe = Pipeline().source("image_bytes").pad_to_size(height=100, width=200)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "pad_to_size")

        try:
            pos_enum = PadPosition(position)
        except ValueError as e:
            valid = [p.value for p in PadPosition]
            msg = f"Invalid position '{position}'. Valid: {valid}"
            raise ValueError(msg) from e

        new = self._clone()
        new._ops.append(
            OpSpec(
                op="pad_to_size",
                params={
                    "height": new._track_expr(height),
                    "width": new._track_expr(width),
                    "position": ParamValue(is_expr=False, value=pos_enum.value),
                    "value": ParamValue(is_expr=False, value=value),
                },
            )
        )
        new._update_shape_hints("pad_to_size", new._ops[-1].params)
        return new

    def letterbox(
        self,
        *,
        height: IntOrExpr,
        width: IntOrExpr,
        value: float = 0.0,
    ) -> "Pipeline":
        """
        Resize image maintaining aspect ratio and pad to exact target size.

        This is a composed operation that:
        1. Resizes the image so it fits within the target dimensions
        2. Pads to reach exact target size with centered positioning

        Domain: buffer → buffer

        Args:
            height: Target height.
            width: Target width.
            value: Fill value for padding (default 0, typically black).

        Returns:
            Self for chaining.

        Example:
            ```python
            >>> # Letterbox any image to 224x224 for VLM input
            >>> pipe = Pipeline().source("image_bytes").letterbox(height=224, width=224)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "letterbox")

        new = self._clone()
        new._ops.append(
            OpSpec(
                op="letterbox",
                params={
                    "height": new._track_expr(height),
                    "width": new._track_expr(width),
                    "value": ParamValue(is_expr=False, value=value),
                },
            )
        )
        new._update_shape_hints("letterbox", new._ops[-1].params)
        return new

    def grayscale(self) -> "Pipeline":
        """
        Convert to grayscale.

        Uses standard luminance formula: 0.299R + 0.587G + 0.114B.
        """
        self._validate_domain(self.DOMAIN_BUFFER, "grayscale")
        new = self._clone()
        new._ops.append(OpSpec(op="grayscale", params={}))
        new._update_output_dtype("grayscale")
        new._update_shape_hints("grayscale", {})
        return new

    def threshold(self, value: IntOrExpr) -> "Pipeline":
        """
        Apply binary threshold.

        Args:
            value: Threshold value (0-255 for u8).
        """
        self._validate_domain(self.DOMAIN_BUFFER, "threshold")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="threshold",
                params={"value": new._track_expr(value)},
            )
        )
        new._update_output_dtype("threshold")
        return new

    def blur(self, sigma: FloatOrExpr) -> "Pipeline":
        """
        Apply Gaussian blur.

        Args:
            sigma: Standard deviation for Gaussian kernel.
        """
        self._validate_domain(self.DOMAIN_BUFFER, "blur")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="blur",
                params={"sigma": new._track_expr(sigma)},
            )
        )
        new._update_output_dtype("blur")
        return new

    def rotate(
        self,
        angle: FloatOrExpr,
        *,
        expand: bool = False,
    ) -> "Pipeline":
        """
        Rotate image by specified angle.

        For angles of 90, 180, or 270 degrees, this uses zero-copy view operations.
        For arbitrary angles, uses bilinear interpolation with allocation.

        Domain: buffer → buffer

        Args:
            angle: Rotation angle in degrees (positive = clockwise).
                Can be a literal float or Polars expression.
            expand: If True, expand output dimensions to fit rotated image.
                If False (default), keep original dimensions (corners may be cropped).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Zero-copy 90-degree rotation
            >>> pipe = Pipeline().source("image_bytes").rotate(90).sink("numpy")
            >>>
            >>> # Arbitrary angle with expansion
            >>> pipe = Pipeline().source("image_bytes").rotate(45, expand=True).sink("numpy")
            >>>
            >>> # Dynamic angle from column
            >>> pipe = Pipeline().source("image_bytes").rotate(pl.col("angle")).sink("numpy")
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "rotate")
        new = self._clone()
        params: dict[str, ParamValue] = {
            "angle": new._track_expr(angle),
            "expand": ParamValue(is_expr=False, value=expand),
        }
        new._ops.append(OpSpec(op="rotate", params=params))
        new._update_output_dtype("rotate")
        new._update_shape_hints("rotate", new._ops[-1].params)
        return new

    def perceptual_hash(
        self,
        algorithm: HashAlgorithm | str = HashAlgorithm.PERCEPTUAL,
        hash_size: int = 64,
    ) -> "Pipeline":
        """
        Compute a perceptual hash fingerprint.

        Args:
            algorithm: "perceptual" (pHash), "average" (aHash), "difference" (dHash).
            hash_size: Number of bits in the hash (must be power of 2).

        Example:
            >>> Pipeline().source("image_bytes").perceptual_hash()
        """
        self._validate_domain(self.DOMAIN_BUFFER, "perceptual_hash")

        # Convert string to enum if needed
        if isinstance(algorithm, str):
            try:
                algorithm = HashAlgorithm(algorithm)
            except ValueError as e:
                valid = [h.value for h in HashAlgorithm]
                msg = f"Invalid algorithm '{algorithm}'. Valid: {valid}"
                raise ValueError(msg) from e

        if hash_size <= 0:
            msg = "hash_size must be a positive integer"
            raise ValueError(msg)

        new = self._clone()
        new._ops.append(
            OpSpec(
                op="perceptual_hash",
                params={
                    "algorithm": ParamValue(is_expr=False, value=algorithm.value),
                    "hash_size": ParamValue(is_expr=False, value=hash_size),
                },
            )
        )
        # Transition to vector domain (fixed-length output)
        new._current_domain = self.DOMAIN_VECTOR
        new._update_output_dtype("perceptual_hash")
        return new

    # --- Contour/Geometry Operations ---

    def rasterize(
        self,
        *,
        width: IntOrExpr | None = None,
        height: IntOrExpr | None = None,
        shape: "LazyPipelineExpr | None" = None,
        fill_value: int = 255,
        background: int = 0,
        anti_alias: bool = False,
    ) -> "Pipeline":
        """
        Rasterize contour to a binary mask.

        Args:
            width: Mask width.
            height: Mask height.
            shape: Match dimensions from another pipeline.
            fill_value: Inside value (default 255).
            background: Outside value (default 0).

        Domain transition: contour → buffer
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "rasterize")
        new = self._clone()

        has_explicit = width is not None or height is not None
        has_shape = shape is not None

        if not has_explicit and not has_shape:
            msg = "Must specify width/height or shape, not neither"
            raise ValueError(msg)
        if has_explicit and has_shape:
            msg = "Specify width/height or shape, not both"
            raise ValueError(msg)

        params: dict[str, ParamValue] = {
            "fill_value": ParamValue(is_expr=False, value=fill_value),
            "background": ParamValue(is_expr=False, value=background),
            "anti_alias": ParamValue(is_expr=False, value=anti_alias),
        }

        if has_explicit:
            if width is None or height is None:
                msg = "Both width and height must be specified"
                raise ValueError(msg)
            params["width"] = new._track_expr(width)
            params["height"] = new._track_expr(height)
        else:
            # 'shape' parameter - store as reference for graph composition
            # This will be resolved during graph execution
            from polars_cv.lazy import LazyPipelineExpr

            if not isinstance(shape, LazyPipelineExpr):
                msg = "'shape' must be a LazyPipelineExpr"
                raise TypeError(msg)
            params["shape_ref"] = ParamValue(is_expr=False, value=shape._node_id)

        new._ops.append(OpSpec(op="rasterize", params=params))
        new._current_domain = self.DOMAIN_BUFFER
        # Rasterize produces a u8 buffer image
        new._output_dtype = "u8"
        return new

    def extract_contours(
        self,
        *,
        mode: str = "external",
        method: str = "simple",
        min_area: float | None = None,
    ) -> "Pipeline":
        """
        Extract contours from binary mask.

        Args:
            mode: "external" (outer only), "tree" (full hierarchy), "all".
            method: "simple" (remove redundant), "none" (all points), "approx".
            min_area: Filter small contours.

        Domain transition: buffer → contour
        """
        self._validate_domain(self.DOMAIN_BUFFER, "extract_contours")
        new = self._clone()

        params: dict[str, ParamValue] = {
            "mode": ParamValue(is_expr=False, value=mode),
            "method": ParamValue(is_expr=False, value=method),
        }

        if min_area is not None:
            params["min_area"] = ParamValue(is_expr=False, value=min_area)

        new._ops.append(OpSpec(op="extract_contours", params=params))
        new._current_domain = self.DOMAIN_CONTOUR
        return new

    # --- Buffer Reduction Operations (buffer → scalar) ---

    def reduce_sum(self) -> "Pipeline":
        """
        Sum all elements in the buffer.

        Domain transition: buffer → scalar
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_sum")
        new = self._clone()
        new._ops.append(OpSpec(op="reduce_sum", params={}))
        new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("reduce_sum")
        return new

    def reduce_popcount(self) -> "Pipeline":
        """
        Count set bits (1s) in the buffer.

        Domain transition: buffer → scalar
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_popcount")
        new = self._clone()
        new._ops.append(OpSpec(op="reduce_popcount", params={}))
        new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("reduce_popcount")
        return new

    def reduce_max(self, axis: int | None = None) -> "Pipeline":
        """
        Reduce buffer by computing the maximum value.

        When axis is None, computes the global maximum across all elements,
        returning a single scalar. When axis is specified, reduces along that
        axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global maximum
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_max()
            >>> df.with_columns(max_val=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Maximum along height axis (returns 1D array per column)
            >>> pipe = Pipeline().source("image_bytes").reduce_max(axis=0)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_max")
        new = self._clone()
        params: dict[str, ParamValue] = {}
        if axis is not None:
            params["axis"] = ParamValue(is_expr=False, value=axis)
        new._ops.append(OpSpec(op="reduce_max", params=params))
        if axis is None:
            new._current_domain = self.DOMAIN_SCALAR
        # axis reduction keeps buffer domain with reduced shape
        new._update_output_dtype("reduce_max")
        return new

    def reduce_min(self, axis: int | None = None) -> "Pipeline":
        """
        Reduce buffer by computing the minimum value.

        When axis is None, computes the global minimum across all elements,
        returning a single scalar. When axis is specified, reduces along that
        axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global minimum
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_min()
            >>> df.with_columns(min_val=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Minimum along width axis
            >>> pipe = Pipeline().source("image_bytes").reduce_min(axis=1)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_min")
        new = self._clone()
        params: dict[str, ParamValue] = {}
        if axis is not None:
            params["axis"] = ParamValue(is_expr=False, value=axis)
        new._ops.append(OpSpec(op="reduce_min", params=params))
        if axis is None:
            new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("reduce_min")
        return new

    def reduce_mean(self, axis: int | None = None) -> "Pipeline":
        """
        Compute arithmetic mean.

        Args:
            axis: Axis to reduce along. If None, computes global mean.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_mean")
        new = self._clone()
        params: dict[str, ParamValue] = {}
        if axis is not None:
            params["axis"] = ParamValue(is_expr=False, value=axis)
        new._ops.append(OpSpec(op="reduce_mean", params=params))
        if axis is None:
            new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("reduce_mean")
        return new

    def reduce_std(self, axis: int | None = None, ddof: int = 0) -> "Pipeline":
        """
        Reduce buffer by computing the standard deviation.

        When axis is None, computes the global standard deviation across all
        elements, returning a single scalar. When axis is specified, reduces
        along that axis, returning a buffer with one fewer dimension.

        Domain transition:
            - axis=None: buffer → scalar
            - axis=N: buffer → buffer (reduced shape)

        Args:
            axis: Axis to reduce along. None for global reduction.
            ddof: Delta degrees of freedom. 0 for population std (default),
                1 for sample std.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Global standard deviation
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_std()
            >>> df.with_columns(std=pl.col("image").cv.pipe(pipe).sink("native"))
            >>>
            >>> # Sample std (ddof=1)
            >>> pipe = Pipeline().source("image_bytes").reduce_std(ddof=1)
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_std")
        new = self._clone()
        params: dict[str, ParamValue] = {
            "ddof": ParamValue(is_expr=False, value=ddof),
        }
        if axis is not None:
            params["axis"] = ParamValue(is_expr=False, value=axis)
        new._ops.append(OpSpec(op="reduce_std", params=params))
        if axis is None:
            new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("reduce_std")
        return new

    def reduce_argmax(self, axis: int) -> "Pipeline":
        """
        Reduce buffer by finding the index of the maximum value along an axis.

        Unlike other reductions, argmax always requires an axis since the global
        argmax would be ambiguous for multi-dimensional arrays.

        Domain transition: buffer → buffer (reduced shape, i64 dtype)

        Args:
            axis: Axis along which to find the maximum index.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Find column with max value per row
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_argmax(axis=1)
            >>> df.with_columns(max_col=pl.col("image").cv.pipe(pipe).sink("list"))
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_argmax")
        new = self._clone()
        params: dict[str, ParamValue] = {
            "axis": ParamValue(is_expr=False, value=axis),
        }
        new._ops.append(OpSpec(op="reduce_argmax", params=params))
        # argmax always returns buffer with reduced shape (indices)
        new._update_output_dtype("reduce_argmax")
        return new

    def reduce_argmin(self, axis: int) -> "Pipeline":
        """
        Reduce buffer by finding the index of the minimum value along an axis.

        Unlike other reductions, argmin always requires an axis since the global
        argmin would be ambiguous for multi-dimensional arrays.

        Domain transition: buffer → buffer (reduced shape, i64 dtype)

        Args:
            axis: Axis along which to find the minimum index.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not buffer.

        Example:
            ```python
            >>> # Find column with min value per row
            >>> pipe = Pipeline().source("image_bytes").grayscale().reduce_argmin(axis=1)
            >>> df.with_columns(min_col=pl.col("image").cv.pipe(pipe).sink("list"))
            ```
        """
        self._validate_domain(self.DOMAIN_BUFFER, "reduce_argmin")
        new = self._clone()
        params: dict[str, ParamValue] = {
            "axis": ParamValue(is_expr=False, value=axis),
        }
        new._ops.append(OpSpec(op="reduce_argmin", params=params))
        # argmin always returns buffer with reduced shape (indices)
        new._update_output_dtype("reduce_argmin")
        return new

    def extract_shape(self) -> "Pipeline":
        """
        Extract buffer shape as a struct {height, width, channels}.

        Domain transition: buffer → vector
        """
        self._validate_domain(self.DOMAIN_BUFFER, "extract_shape")
        new = self._clone()
        new._ops.append(OpSpec(op="extract_shape", params={}))
        new._current_domain = self.DOMAIN_VECTOR
        new._update_output_dtype("extract_shape")
        return new

    def histogram(
        self,
        bins: int = 256,
        range: tuple[float, float] | None = None,
        output: str = "counts",
    ) -> "Pipeline":
        """
        Compute pixel value histogram.

        Args:
            bins: Number of bins (default 256).
            range: (min, max) tuple. Auto-detected if None.
            output: "counts" (bin counts), "normalized" (sum to 1.0),
                    "quantized" (pixel indices), "edges" (bin edges).

        Example:
            >>> Pipeline().source("image_bytes").grayscale().histogram(bins=8)
        """
        self._validate_domain(self.DOMAIN_BUFFER, "histogram")

        # Validate output mode
        try:
            output_mode = HistogramOutput(output)
        except ValueError as e:
            valid = [o.value for o in HistogramOutput]
            msg = f"Invalid histogram output mode '{output}'. Valid: {valid}"
            raise ValueError(msg) from e

        new = self._clone()

        params: dict[str, ParamValue] = {
            "bins": ParamValue(is_expr=False, value=bins),
            "output": ParamValue(is_expr=False, value=output_mode.value),
        }

        if range is not None:
            params["range_min"] = ParamValue(is_expr=False, value=range[0])
            params["range_max"] = ParamValue(is_expr=False, value=range[1])

        new._ops.append(OpSpec(op="histogram", params=params))

        # Domain transition depends on output mode
        if output_mode == HistogramOutput.QUANTIZED:
            # Quantized preserves the buffer domain
            new._current_domain = self.DOMAIN_BUFFER
            new._output_dtype = "u32"
        else:
            # All other modes return a vector
            new._current_domain = self.DOMAIN_VECTOR
            if output_mode == HistogramOutput.COUNTS:
                new._output_dtype = "u64"
            else:  # normalized or edges
                new._output_dtype = "f64"

        return new

    # --- Contour Measure Operations (contour → scalar/vector) ---

    def area(self, *, signed: bool = False) -> "Pipeline":
        """
        Compute the area of the contour using the Shoelace formula.

        Domain transition: contour → scalar

        Args:
            signed: If True, return signed area (negative for CW winding).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "area")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="contour_area",
                params={"signed": ParamValue(is_expr=False, value=signed)},
            )
        )
        new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("contour_area")
        return new

    def perimeter(self) -> "Pipeline":
        """
        Compute the perimeter (arc length) of the contour.

        Domain transition: contour → scalar

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "perimeter")
        new = self._clone()
        new._ops.append(OpSpec(op="contour_perimeter", params={}))
        new._current_domain = self.DOMAIN_SCALAR
        new._update_output_dtype("contour_perimeter")
        return new

    def centroid(self) -> "Pipeline":
        """
        Compute the centroid (center of mass) of the contour.

        Domain transition: contour → vector (returns [x, y])

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "centroid")
        new = self._clone()
        new._ops.append(OpSpec(op="contour_centroid", params={}))
        new._current_domain = self.DOMAIN_VECTOR
        new._update_output_dtype("contour_centroid")
        return new

    def bounding_box(self) -> "Pipeline":
        """
        Compute the axis-aligned bounding box of the contour.

        Domain transition: contour → vector (returns [x, y, width, height])

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "bounding_box")
        new = self._clone()
        new._ops.append(OpSpec(op="contour_bounding_box", params={}))
        new._current_domain = self.DOMAIN_VECTOR
        new._update_output_dtype("contour_bounding_box")
        return new

    # --- Contour Transform Operations (contour → contour) ---

    def translate(self, *, dx: FloatOrExpr, dy: FloatOrExpr) -> "Pipeline":
        """
        Translate the contour by an offset.

        Domain: contour → contour

        Args:
            dx: X offset (horizontal translation).
            dy: Y offset (vertical translation).

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "translate")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="contour_translate",
                params={
                    "dx": new._track_expr(dx),
                    "dy": new._track_expr(dy),
                },
            )
        )
        return new

    def scale_contour(
        self,
        *,
        sx: FloatOrExpr,
        sy: FloatOrExpr,
    ) -> "Pipeline":
        """
        Scale the contour relative to its centroid.

        Domain: contour → contour

        Args:
            sx: X scale factor.
            sy: Y scale factor.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "scale_contour")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="contour_scale",
                params={
                    "sx": new._track_expr(sx),
                    "sy": new._track_expr(sy),
                },
            )
        )
        return new

    def simplify(self, *, tolerance: FloatOrExpr) -> "Pipeline":
        """
        Simplify the contour using Douglas-Peucker algorithm.

        Domain: contour → contour

        Args:
            tolerance: Maximum distance from original contour.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "simplify")
        new = self._clone()
        new._ops.append(
            OpSpec(
                op="contour_simplify",
                params={"tolerance": new._track_expr(tolerance)},
            )
        )
        return new

    def convex_hull(self) -> "Pipeline":
        """
        Compute the convex hull of the contour.

        Domain: contour → contour

        Returns:
            Self for chaining.

        Raises:
            ValueError: If current domain is not contour.
        """
        self._validate_domain(self.DOMAIN_CONTOUR, "convex_hull")
        new = self._clone()
        new._ops.append(OpSpec(op="contour_convex_hull", params={}))
        return new

    # --- Sink (required, ends the chain) ---

    def sink(
        self,
        format: str,
        *,
        quality: int = 85,
        shape: list[int] | None = None,
    ) -> "Pipeline":
        """
        Define output format.

        Args:
            format: "numpy", "png", "jpeg", "blob", "array", "list", "native".
            quality: JPEG quality (1-100).
            shape: Required for "array" format.
        """
        new = self._clone()

        try:
            fmt = SinkFormat(format)
        except ValueError as e:
            valid = [f.value for f in SinkFormat]
            msg = f"Invalid sink format '{format}'. Valid: {valid}"
            raise ValueError(msg) from e

        if fmt == SinkFormat.ARRAY and shape is None:
            # Check if shape is deterministic in the pipeline
            if not new._shape_hints.has_all_dims():
                msg = (
                    "shape is required for 'array' sink format when output shape is not deterministic. "
                    "Provide 'shape' in .sink() or use .resize()/.assert_shape() earlier."
                )
                raise ValueError(msg)

        new._sink = SinkSpec(format=fmt, quality=quality, shape=shape)

        return new

    # --- Validation ---

    def validate(self) -> None:
        """
        Validate that the pipeline is complete and well-formed.

        Raises:
            ValueError: If pipeline is incomplete or invalid.
        """
        if self._source is None:
            msg = "Pipeline must have a source. Call .source() first."
            raise ValueError(msg)
        if self._sink is None:
            msg = "Pipeline must have a sink. Call .sink() at the end."
            raise ValueError(msg)

    def has_sink(self) -> bool:
        """
        Check if the pipeline has a sink defined.

        Returns:
            True if the pipeline has a sink defined.
        """
        return self._sink is not None

    def has_source(self) -> bool:
        """
        Check if the pipeline has a source defined.

        Returns:
            True if the pipeline has a source defined.
        """
        return self._source is not None

    # --- Graph Conversion ---

    def to_graph(self, column: pl.Expr | None = None) -> "PipelineGraph":
        """
        Convert this linear pipeline to a graph representation.

        This is the unified execution path - all pipelines are converted to
        graphs before execution. A Pipeline becomes a single node in the graph.

        For multi-output with intermediate checkpoints, use LazyPipelineExpr
        composition with .pipe() and .alias() instead.

        Args:
            column: The input column expression. If None, must be set later
                via graph.set_root_column().

        Returns:
            PipelineGraph representation of this pipeline.

        Example:
            ```python
            >>> pipe = Pipeline().source("image_bytes").resize(100, 200).sink("numpy")
            >>> graph = pipe.to_graph(pl.col("image"))
            >>> expr = graph.to_expr()
            ```
        """
        from polars_cv._graph import PipelineGraph

        graph = PipelineGraph()

        # Create single node with all operations
        node_id = "_node_0"
        # Create a sub-pipeline with source and all ops (no sink - handled separately)
        sub_pipe = self._create_sub_pipeline(0, len(self._ops))
        graph.add_node(
            node_id=node_id,
            pipeline=sub_pipe,
            column=column,
            upstream=[],
            alias="_output",  # Implicit terminal alias
        )
        graph._alias_to_node["_output"] = node_id

        return graph

    def _create_sub_pipeline(
        self,
        start_op: int,
        end_op: int,
        source_format: str | None = None,
    ) -> "Pipeline":
        """
        Create a sub-pipeline with a subset of operations.

        Args:
            start_op: Starting operation index (inclusive).
            end_op: Ending operation index (exclusive).
            source_format: Override source format (e.g., "blob" for non-root nodes).

        Returns:
            New Pipeline with the specified operations.
        """
        sub = Pipeline()

        if source_format is not None:
            # Non-root node: source is blob (receives from upstream)
            sub._source = SourceSpec(format=SourceFormat(source_format))
        else:
            # Root node: use original source
            sub._source = self._source

        sub._shape_hints = self._shape_hints
        sub._ops = self._ops[start_op:end_op]
        sub._expr_refs = self._expr_refs.copy()

        # Compute the correct domain and dtype for this subset of operations
        # We need to compute from the beginning up to end_op to get correct state
        ops_to_compute = self._ops[0:end_op]
        domain, dtype, ndim = Pipeline._compute_output_domain_dtype_ndim(
            ops_to_compute,
            initial_dtype=self._output_dtype,
            initial_ndim=self._expected_ndim,
        )
        sub._current_domain = domain
        sub._output_dtype = dtype
        sub._expected_ndim = ndim
        sub._auto_infer_from_input = self._auto_infer_from_input

        return sub

    # --- Graph Composition Support ---

    def _add_binary_op(
        self,
        op: str,
        other_node_id: str,
        **kwargs,
    ) -> None:
        """
        Add a binary operation referencing another node.

        This is used internally by LazyPipelineExpr composition.

        Args:
            op: Operation name (e.g., "add", "multiply", "apply_mask").
            other_node_id: The node ID of the other operand.
            **kwargs: Additional operation parameters.
        """
        params: dict[str, ParamValue] = {
            "other_node": ParamValue(is_expr=False, value=other_node_id),
        }
        for key, value in kwargs.items():
            params[key] = ParamValue(is_expr=False, value=value)

        self._ops.append(OpSpec(op=op, params=params))

    def _to_spec_dict(self) -> dict:
        """
        Convert pipeline to specification dictionary (without sink).

        Used for graph serialization where sink is handled separately.

        Returns:
            Dictionary with source, shape_hints, ops, domain, and output_dtype.
        """
        spec: dict = {
            "source": self._source.to_dict() if self._source else None,
            "ops": [op.to_dict() for op in self._ops],
            "domain": self._current_domain,
            "output_dtype": self._output_dtype,
        }

        if self._shape_hints.has_any():
            spec["shape_hints"] = self._shape_hints.to_dict()

        return spec

    # --- Serialization ---

    def _to_json(self) -> str:
        """
        Serialize pipeline to JSON for the Rust plugin.

        Returns:
            JSON string representation of the pipeline.

        Raises:
            ValueError: If pipeline is incomplete.
        """
        self.validate()

        spec: dict = {
            "source": self._source.to_dict() if self._source else None,
            "ops": [op.to_dict() for op in self._ops],
        }

        if self._sink is not None:
            spec["sink"] = self._sink.to_dict()

        if self._shape_hints.has_any():
            spec["shape_hints"] = self._shape_hints.to_dict()

        return json.dumps(spec)

    def _get_expr_columns(self) -> list[pl.Expr]:
        """
        Get all expression columns referenced by this pipeline.

        Returns:
            List of Polars expressions that need to be passed to the plugin.
        """
        return self._expr_refs.copy()

    # --- Repr ---

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        parts = []
        if self._source:
            parts.append(f"source({self._source.format.value!r})")
        if self._shape_hints.has_any():
            hints = []
            if self._shape_hints.height:
                hints.append(f"height={self._shape_hints.height.value}")
            if self._shape_hints.width:
                hints.append(f"width={self._shape_hints.width.value}")
            if self._shape_hints.channels:
                hints.append(f"channels={self._shape_hints.channels.value}")
            parts.append(f"assert_shape({', '.join(hints)})")
        for op in self._ops:
            params_str = ", ".join(f"{k}={v.value}" for k, v in op.params.items())
            parts.append(f"{op.op}({params_str})")
        if self._sink:
            parts.append(f"sink({self._sink.format.value!r})")

        return f"Pipeline().{'.'.join(parts)}" if parts else "Pipeline()"
        return f"Pipeline().{'.'.join(parts)}" if parts else "Pipeline()"
        return f"Pipeline().{'.'.join(parts)}" if parts else "Pipeline()"
