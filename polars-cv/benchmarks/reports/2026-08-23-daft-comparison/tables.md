# Daft vs polars-cv — benchmark tables

### Benchmark coverage

Cells completed out of 40 (20 single ops x 2 image sizes).

| framework | cells | coverage |
|---|---:|---:|
| pcv-eager | 40 | 100% |
| pcv-stream | 40 | 100% |
| daft | 6 | 15% |
| daft-udf | 40 | 100% |
| opencv | 40 | 100% |
| pillow | 36 | 90% |

### Head-to-head on Daft's native operations

Throughput ratios on the three single ops Daft implements with its own
expressions — the only cells where both engines run their own kernels on
comparable work.

| op | size | pcv-eager ÷ daft | pcv-stream ÷ daft | daft ÷ opencv |
|---|---:|---:|---:|---:|
| resize | 256 | 1.67x | 5.61x | 1.23x |
| resize | 512 | 1.84x | 7.45x | 1.53x |
| grayscale | 256 | 0.97x | 3.24x | 0.45x |
| grayscale | 512 | 0.63x | 1.70x | 0.29x |
| crop_center | 256 | 0.84x | 2.00x | 0.28x |
| crop_center | 512 | 0.66x | 1.79x | 0.17x |
| **geomean** | | **1.01x** | **3.06x** | **0.47x** |

### What Daft's batch-UDF path costs

`daft-udf` calls the very same OpenCV kernels as the `opencv` adapter, so
this ratio isolates Daft's UDF machinery against a plain single-threaded
Python loop. Below 1.00x, the dataframe engine is losing on ops it has to
hand back to Python.

| op | size | daft-udf ÷ opencv | pcv-stream ÷ opencv |
|---|---:|---:|---:|
| adjust_brightness | 256 | 0.46x | 1.74x |
| adjust_contrast | 256 | 1.50x | 9.48x |
| blur | 256 | 0.63x | 1.09x |
| canny | 256 | 0.35x | 0.44x |
| crop_center | 256 | 0.27x | 0.55x |
| dilate | 256 | 0.30x | 1.66x |
| erode | 256 | 0.30x | 1.57x |
| flip_horizontal | 256 | 0.30x | 1.00x |
| flip_vertical | 256 | 0.30x | 0.87x |
| grayscale | 256 | 0.43x | 1.46x |
| histogram_equalize | 256 | 0.33x | 1.06x |
| invert | 256 | 0.25x | 2.21x |
| normalize | 256 | 0.44x | 2.81x |
| pad | 256 | 0.25x | 2.24x |
| resize | 256 | 1.18x | 6.92x |
| rotate_45 | 256 | 0.49x | 0.85x |
| rotate_90 | 256 | 0.63x | 4.90x |
| sharpen | 256 | 0.53x | 2.12x |
| sobel_x | 256 | 0.45x | 3.45x |
| threshold | 256 | 0.25x | 1.58x |
| adjust_brightness | 512 | 0.52x | 2.12x |
| adjust_contrast | 512 | 0.81x | 7.14x |
| blur | 512 | 0.31x | 0.63x |
| canny | 512 | 0.40x | 0.33x |
| crop_center | 512 | 0.16x | 0.30x |
| dilate | 512 | 0.42x | 1.40x |
| erode | 512 | 0.38x | 1.32x |
| flip_horizontal | 512 | 0.27x | 0.96x |
| flip_vertical | 512 | 0.25x | 0.92x |
| grayscale | 512 | 0.29x | 0.49x |
| histogram_equalize | 512 | 0.53x | 1.09x |
| invert | 512 | 0.26x | 2.96x |
| normalize | 512 | 0.35x | 2.88x |
| pad | 512 | 0.32x | 2.60x |
| resize | 512 | 1.58x | 11.41x |
| rotate_45 | 512 | 0.38x | 0.44x |
| rotate_90 | 512 | 0.35x | 1.28x |
| sharpen | 512 | 0.67x | 1.91x |
| sobel_x | 512 | 0.37x | 3.74x |
| threshold | 512 | 0.18x | 0.68x |
| **geomean** | | **0.40x** | **1.55x** |

### Single operations

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|
| adjust_brightness | 256 | 1,961 | 6,457 | — | 1,701 | 3,710 | 1,985 |
| adjust_contrast | 256 | 1,493 | 5,131 | — | 810 | 541 | 1,452 |
| blur | 256 | 958 | 3,209 | — | 1,863 | 2,938 | 522 |
| canny | 256 | 1,187 | 3,899 | — | 3,029 | 8,766 | — |
| crop_center | 256 | 8,217 | 19,645 | 9,833 | 9,604 | 35,745 | 13,794 |
| dilate | 256 | 5,267 | 18,116 | — | 3,229 | 10,884 | 791 |
| erode | 256 | 5,420 | 17,580 | — | 3,365 | 11,186 | 829 |
| flip_horizontal | 256 | 2,678 | 9,319 | — | 2,828 | 9,284 | 7,193 |
| flip_vertical | 256 | 2,674 | 8,446 | — | 2,904 | 9,763 | 8,228 |
| grayscale | 256 | 6,395 | 21,381 | 6,590 | 6,256 | 14,633 | 7,216 |
| histogram_equalize | 256 | 2,805 | 9,226 | — | 2,877 | 8,733 | 3,809 |
| invert | 256 | 7,780 | 21,886 | — | 2,425 | 9,893 | 4,465 |
| normalize | 256 | 1,802 | 5,353 | — | 832 | 1,903 | 2,130 |
| pad | 256 | 5,488 | 16,974 | — | 1,913 | 7,575 | 6,833 |
| resize | 256 | 3,705 | 12,434 | 2,216 | 2,128 | 1,797 | 1,395 |
| rotate_45 | 256 | 373 | 1,441 | — | 827 | 1,704 | 2,371 |
| rotate_90 | 256 | 2,659 | 8,862 | — | 1,138 | 1,807 | 4,787 |
| sharpen | 256 | 1,270 | 4,385 | — | 1,102 | 2,073 | 880 |
| sobel_x | 256 | 2,357 | 8,126 | — | 1,064 | 2,357 | — |
| threshold | 256 | 6,037 | 20,804 | — | 3,363 | 13,190 | 4,692 |
| adjust_brightness | 512 | 473 | 1,665 | — | 408 | 784 | 546 |
| adjust_contrast | 512 | 374 | 1,344 | — | 152 | 188 | 387 |
| blur | 512 | 255 | 993 | — | 493 | 1,571 | 125 |
| canny | 512 | 316 | 1,211 | — | 1,479 | 3,666 | — |
| crop_center | 512 | 6,418 | 17,506 | 9,761 | 9,565 | 58,691 | 4,888 |
| dilate | 512 | 1,389 | 5,506 | — | 1,664 | 3,922 | 162 |
| erode | 512 | 1,401 | 5,434 | — | 1,557 | 4,128 | 168 |
| flip_horizontal | 512 | 666 | 2,329 | — | 668 | 2,438 | 1,969 |
| flip_vertical | 512 | 664 | 2,430 | — | 655 | 2,633 | 2,392 |
| grayscale | 512 | 1,603 | 4,314 | 2,539 | 2,562 | 8,822 | 1,742 |
| histogram_equalize | 512 | 714 | 2,831 | — | 1,385 | 2,595 | 1,082 |
| invert | 512 | 1,848 | 7,737 | — | 691 | 2,618 | 1,460 |
| normalize | 512 | 422 | 1,363 | — | 167 | 473 | 542 |
| pad | 512 | 1,504 | 5,752 | — | 708 | 2,216 | 1,757 |
| resize | 512 | 2,244 | 9,059 | 1,217 | 1,253 | 794 | 585 |
| rotate_45 | 512 | 93 | 361 | — | 312 | 826 | 495 |
| rotate_90 | 512 | 608 | 2,130 | — | 578 | 1,664 | 1,180 |
| sharpen | 512 | 333 | 1,183 | — | 412 | 620 | 208 |
| sobel_x | 512 | 526 | 2,412 | — | 237 | 645 | — |
| threshold | 512 | 1,555 | 5,774 | — | 1,519 | 8,453 | 1,314 |

### Multi-operation pipelines

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|
| heavy_pipeline | 256 | 1,394 | 4,670 | — | 697 | 2,821 | 859 |
| imagenet_preprocess | 256 | 1,461 | 5,457 | — | 908 | 2,315 | 2,444 |
| light_pipeline | 256 | 1,660 | 5,532 | — | 671 | 1,054 | 961 |
| medical_pipeline | 256 | 893 | 3,306 | — | 562 | 2,681 | 780 |
| medium_pipeline | 256 | 1,130 | 3,872 | — | 651 | 1,941 | 1,785 |
| heavy_pipeline | 512 | 973 | 3,429 | — | 473 | 1,638 | 342 |
| imagenet_preprocess | 512 | 981 | 3,616 | — | 620 | 1,557 | 464 |
| light_pipeline | 512 | 1,228 | 4,472 | — | 607 | 623 | 504 |
| medical_pipeline | 512 | 798 | 2,756 | — | 314 | 1,134 | 847 |
| medium_pipeline | 512 | 776 | 2,515 | — | 428 | 1,353 | 422 |

### End-to-end file workflows

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|
| e2e_augmentation_workflow | 256 | 722 | 1,401 | — | 587 | 1,113 | 882 |
| e2e_basic_preprocess | 256 | 1,044 | 1,783 | — | 665 | 800 | 600 |
| e2e_imagenet_workflow | 256 | 916 | 1,716 | — | 762 | 1,314 | 1,032 |
| e2e_augmentation_workflow | 512 | 537 | 1,142 | — | 303 | 532 | 262 |
| e2e_basic_preprocess | 512 | 727 | 1,593 | — | 402 | 356 | 289 |
| e2e_imagenet_workflow | 512 | 637 | 1,413 | — | 393 | 557 | 274 |

