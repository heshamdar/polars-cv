# Migrating from 0.38 to 0.39

0.39 changes one default shape and corrects how `apply_mask` scales a mask.
This page lists what to check coming from 0.38; the
[changelog](../changelog.md) has the full list.

## Shapes that change

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
