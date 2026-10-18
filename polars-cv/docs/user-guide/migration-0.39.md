# Migrating from 0.38 to 0.39

0.39 stops binary operations broadcasting, changes one default shape and
corrects how `apply_mask` scales a mask.
This page lists what to check coming from 0.38; the
[changelog](../changelog.md) has the full list.

## Shapes that change

**Binary operations no longer broadcast.** `add`, `subtract`, `multiply`,
`divide`, `blend`, `maximum`, `minimum` and the bitwise ops need operands of
the same shape, and refuse any other pair where it is written (or per row,
where a size is known only then). 0.38 broadcast NumPy-style from the last
axis, so a `[H, W, 1]` operand stretched silently over `[H, W, 3]`, and
`[H, W, 1]` against `[H, W]` became `[H, W, W]`. Change a rank with the new
`squeeze(axis)` / `expand_dims(axis)`:

```python
gray = img.pipe(Pipeline().grayscale())        # [H, W, 1]
plane = img.channel_select(0, keepdims=False)  # [H, W]
plane.multiply(gray.squeeze(2))                # both [H, W]; refused unsqueezed
```

`apply_mask` still spreads a single-channel mask over an image's channels.
See [Binary Operations](composition/binary-ops.md).

**`channel_select(i)` keeps the channel axis.** It returns `[H, W, 1]`, the
shape `grayscale` and `rasterize` return, instead of `[H, W]`. A selected
channel now combines with those without reshaping:

```python
gray = img.pipe(Pipeline().grayscale())          # [H, W, 1]
brightest = gray.maximum(img.channel_select(2))   # [H, W, 1] too
```

Where you need the 2-D result (a `[H, W]` array sink, matplotlib's `imshow`,
an operand that is `[H, W]`), pass `keepdims=False`:

```python
Pipeline().source("image_bytes").channel_select(0, keepdims=False)  # [H, W]
```

A `[H, W]` input is returned as it is either way. `keepdims` fixes the
output's rank, so it is a literal, not a per-row expression.

## Results that change

**`apply_mask` reads the mask on the mask's own scale.** A u8 mask weighs each
pixel by `m / 255` and a float mask by `m` in `[0, 1]`, whatever the image's
dtype, and the result keeps the image's dtype. 0.38 read the mask on the
image's scale: a u8 mask (a `rasterize` or `threshold` result) multiplied a
float image by 255 and darkened a u16 one 257-fold, and a float mask was
truncated to 0 or 1 on a u8 image. A u8 image under a u8 mask is unchanged. A
mask of another dtype is now refused; cast or scale it first.
