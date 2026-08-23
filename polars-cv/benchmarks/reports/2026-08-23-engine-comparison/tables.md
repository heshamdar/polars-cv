# Daft vs polars-cv — benchmark tables

### Benchmark coverage

Cells completed out of 40 (20 single ops x 2 image sizes).

| framework | cells | coverage |
|---|---:|---:|
| pcv-eager | 40 | 100% |
| pcv-stream | 40 | 100% |
| daft | 6 | 15% |
| daft-udf | 40 | 100% |
| pxt | 18 | 45% |
| pxt-udf | 40 | 100% |
| opencv | 40 | 100% |
| pillow | 36 | 90% |

### Head-to-head on natively-expressible operations

Throughput ratios on the ops each engine implements with its own
expressions — the cells where both sides run comparable work. Daft covers
three of the twenty; Pixeltable covers nine.

| op | size | pcv-stream ÷ daft | pcv-stream ÷ pxt | pcv-eager ÷ pxt |
|---|---:|---:|---:|---:|
| adjust_brightness | 256 | — | 5.29x | 1.85x |
| crop_center | 256 | 3.02x | 14.55x | 5.78x |
| flip_horizontal | 256 | — | 4.77x | 2.08x |
| flip_vertical | 256 | — | 5.60x | 2.12x |
| grayscale | 256 | 3.77x | 13.13x | 4.69x |
| invert | 256 | — | 18.51x | 13.68x |
| resize | 256 | 7.70x | 21.99x | 5.21x |
| rotate_90 | 256 | — | 5.88x | 2.29x |
| threshold | 256 | — | 14.14x | 5.26x |
| adjust_brightness | 512 | — | 2.76x | 1.19x |
| crop_center | 512 | 2.14x | 27.27x | 10.47x |
| flip_horizontal | 512 | — | 5.42x | 1.53x |
| flip_vertical | 512 | — | 3.73x | 1.32x |
| grayscale | 512 | 4.11x | 11.97x | 3.86x |
| invert | 512 | — | 18.88x | 4.83x |
| resize | 512 | 7.62x | 37.94x | 11.23x |
| rotate_90 | 512 | — | 5.45x | 1.51x |
| threshold | 512 | — | 15.36x | 4.16x |
| **geomean** | | **4.25x** | **10.00x** | **3.50x** |

### What Daft's batch-UDF path costs

`daft-udf` calls the very same OpenCV kernels as the `opencv` adapter, so
this ratio isolates Daft's UDF machinery against a plain single-threaded
Python loop. Below 1.00x, the dataframe engine is losing on ops it has to
hand back to Python.

| op | size | daft-udf ÷ opencv | pxt-udf ÷ opencv | pcv-stream ÷ opencv |
|---|---:|---:|---:|---:|
| adjust_brightness | 256 | 0.50x | 0.31x | 1.60x |
| adjust_contrast | 256 | 0.64x | 0.40x | 3.92x |
| blur | 256 | 0.48x | 0.24x | 0.83x |
| canny | 256 | 0.37x | 0.13x | 0.45x |
| crop_center | 256 | 0.13x | 0.03x | 0.39x |
| dilate | 256 | 0.34x | 0.11x | 1.21x |
| erode | 256 | 0.31x | 0.10x | 1.08x |
| flip_horizontal | 256 | 0.20x | 0.10x | 0.51x |
| flip_vertical | 256 | 0.21x | 0.10x | 0.56x |
| grayscale | 256 | 0.25x | 0.08x | 1.02x |
| histogram_equalize | 256 | 0.36x | 0.13x | 0.88x |
| invert | 256 | 0.22x | 0.08x | 1.56x |
| normalize | 256 | 0.16x | 0.44x | 1.98x |
| pad | 256 | 0.31x | 0.15x | 2.04x |
| resize | 256 | 1.12x | 0.38x | 8.50x |
| rotate_45 | 256 | 0.55x | 0.41x | 0.87x |
| rotate_90 | 256 | 0.63x | 0.65x | 3.84x |
| sharpen | 256 | 0.59x | 0.33x | 1.75x |
| sobel_x | 256 | 0.36x | 0.41x | 3.28x |
| threshold | 256 | 0.27x | 0.08x | 1.13x |
| adjust_brightness | 512 | 0.43x | 0.46x | 1.25x |
| adjust_contrast | 512 | 0.45x | 0.38x | 3.01x |
| blur | 512 | 0.27x | 0.20x | 0.59x |
| canny | 512 | 0.30x | 0.11x | 0.29x |
| crop_center | 512 | 0.14x | 0.01x | 0.28x |
| dilate | 512 | 0.26x | 0.09x | 1.06x |
| erode | 512 | 0.26x | 0.10x | 0.93x |
| flip_horizontal | 512 | 0.18x | 0.16x | 0.77x |
| flip_vertical | 512 | 0.18x | 0.16x | 0.58x |
| grayscale | 512 | 0.12x | 0.04x | 0.52x |
| histogram_equalize | 512 | 0.33x | 0.13x | 0.86x |
| invert | 512 | 0.17x | 0.11x | 2.02x |
| normalize | 512 | 0.17x | 0.52x | 1.92x |
| pad | 512 | 0.20x | 0.14x | 2.22x |
| resize | 512 | 1.52x | 0.30x | 11.06x |
| rotate_45 | 512 | 0.30x | 0.28x | 0.45x |
| rotate_90 | 512 | 0.28x | 0.24x | 1.32x |
| sharpen | 512 | 0.43x | 0.35x | 1.63x |
| sobel_x | 512 | 0.23x | 0.43x | 2.79x |
| threshold | 512 | 0.13x | 0.04x | 0.61x |
| **geomean** | | **0.30x** | **0.16x** | **1.21x** |

### Single operations

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | pxt | pxt-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adjust_brightness | 256 | 2,252 | 6,454 | — | 2,026 | 1,220 | 1,233 | 4,030 | 2,082 |
| adjust_contrast | 256 | 1,696 | 5,129 | — | 836 | — | 528 | 1,307 | 1,446 |
| blur | 256 | 1,054 | 3,380 | — | 1,950 | — | 966 | 4,069 | 527 |
| canny | 256 | 1,229 | 4,314 | — | 3,495 | — | 1,257 | 9,517 | — |
| crop_center | 256 | 9,760 | 24,556 | 8,131 | 8,334 | 1,688 | 1,701 | 63,579 | 15,560 |
| dilate | 256 | 5,523 | 15,795 | — | 4,427 | — | 1,383 | 13,011 | 729 |
| erode | 256 | 5,792 | 14,051 | — | 4,076 | — | 1,358 | 12,964 | 754 |
| flip_horizontal | 256 | 3,228 | 7,397 | — | 2,957 | 1,550 | 1,496 | 14,473 | 7,663 |
| flip_vertical | 256 | 3,188 | 8,418 | — | 3,190 | 1,504 | 1,566 | 14,924 | 8,933 |
| grayscale | 256 | 7,397 | 20,714 | 5,496 | 5,026 | 1,577 | 1,593 | 20,305 | 7,808 |
| histogram_equalize | 256 | 2,924 | 9,372 | — | 3,845 | — | 1,337 | 10,638 | 3,968 |
| invert | 256 | 16,911 | 22,881 | — | 3,218 | 1,236 | 1,152 | 14,657 | 4,646 |
| normalize | 256 | 2,023 | 3,869 | — | 321 | — | 871 | 1,959 | 2,104 |
| pad | 256 | 10,582 | 15,919 | — | 2,461 | — | 1,173 | 7,821 | 7,071 |
| resize | 256 | 3,450 | 14,563 | 1,891 | 1,927 | 662 | 647 | 1,714 | 1,403 |
| rotate_45 | 256 | 392 | 1,454 | — | 910 | — | 678 | 1,663 | 2,427 |
| rotate_90 | 256 | 3,038 | 7,801 | — | 1,279 | 1,327 | 1,323 | 2,030 | 5,014 |
| sharpen | 256 | 1,424 | 3,948 | — | 1,334 | — | 742 | 2,255 | 895 |
| sobel_x | 256 | 3,563 | 8,818 | — | 976 | — | 1,098 | 2,687 | — |
| threshold | 256 | 6,917 | 18,590 | — | 4,373 | 1,315 | 1,331 | 16,449 | 4,922 |
| adjust_brightness | 512 | 479 | 1,112 | — | 384 | 403 | 406 | 888 | 536 |
| adjust_contrast | 512 | 373 | 1,006 | — | 151 | — | 126 | 334 | 387 |
| blur | 512 | 267 | 934 | — | 435 | — | 312 | 1,594 | 128 |
| canny | 512 | 316 | 1,187 | — | 1,213 | — | 440 | 4,084 | — |
| crop_center | 512 | 6,428 | 16,744 | 7,817 | 8,185 | 614 | 609 | 59,297 | 5,862 |
| dilate | 512 | 1,424 | 5,023 | — | 1,225 | — | 445 | 4,727 | 149 |
| erode | 512 | 1,548 | 4,321 | — | 1,198 | — | 445 | 4,642 | 153 |
| flip_horizontal | 512 | 677 | 2,402 | — | 571 | 444 | 516 | 3,139 | 2,002 |
| flip_vertical | 512 | 745 | 2,103 | — | 664 | 564 | 560 | 3,602 | 2,510 |
| grayscale | 512 | 2,037 | 6,313 | 1,537 | 1,447 | 527 | 529 | 12,047 | 1,887 |
| histogram_equalize | 512 | 740 | 2,638 | — | 1,019 | — | 406 | 3,067 | 1,113 |
| invert | 512 | 1,897 | 7,411 | — | 612 | 392 | 402 | 3,663 | 1,454 |
| normalize | 512 | 460 | 932 | — | 83 | — | 250 | 485 | 527 |
| pad | 512 | 1,638 | 5,521 | — | 500 | — | 350 | 2,487 | 1,746 |
| resize | 512 | 2,690 | 9,086 | 1,193 | 1,245 | 239 | 243 | 822 | 599 |
| rotate_45 | 512 | 93 | 358 | — | 235 | — | 218 | 792 | 489 |
| rotate_90 | 512 | 637 | 2,306 | — | 492 | 423 | 418 | 1,741 | 1,207 |
| sharpen | 512 | 339 | 1,150 | — | 301 | — | 250 | 705 | 221 |
| sobel_x | 512 | 659 | 1,995 | — | 162 | — | 310 | 714 | — |
| threshold | 512 | 1,668 | 6,157 | — | 1,327 | 401 | 396 | 10,057 | 1,389 |

### Multi-operation pipelines

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | pxt | pxt-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heavy_pipeline | 256 | 1,433 | 4,891 | — | 668 | — | 816 | 3,340 | 833 |
| imagenet_preprocess | 256 | 1,736 | 4,755 | — | 905 | — | 882 | 2,811 | 2,459 |
| light_pipeline | 256 | 1,530 | 4,700 | — | 580 | — | 486 | 1,091 | 975 |
| medical_pipeline | 256 | 993 | 3,155 | — | 515 | — | 926 | 2,840 | 795 |
| medium_pipeline | 256 | 1,262 | 3,478 | — | 560 | — | 761 | 2,193 | 1,835 |
| heavy_pipeline | 512 | 982 | 3,270 | — | 478 | — | 195 | 1,687 | 326 |
| imagenet_preprocess | 512 | 1,187 | 2,835 | — | 577 | — | 214 | 1,665 | 472 |
| light_pipeline | 512 | 1,383 | 4,123 | — | 525 | — | 225 | 643 | 497 |
| medical_pipeline | 512 | 1,094 | 2,788 | — | 262 | — | 387 | 1,271 | 870 |
| medium_pipeline | 512 | 948 | 2,844 | — | 400 | — | 204 | 1,438 | 420 |

### End-to-end file workflows

Throughput, images/second (higher is better).

| op | size | pcv-eager | pcv-stream | daft | daft-udf | pxt | pxt-udf | opencv | pillow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e2e_augmentation_workflow | 256 | 888 | 2,251 | — | 510 | — | 362 | 1,476 | 1,052 |
| e2e_basic_preprocess | 256 | 931 | 1,711 | — | 646 | — | 282 | 935 | 706 |
| e2e_imagenet_workflow | 256 | 1,190 | 3,023 | — | 755 | — | 389 | 1,840 | 1,262 |
| e2e_augmentation_workflow | 512 | 596 | 1,461 | — | 284 | — | 125 | 570 | 268 |
| e2e_basic_preprocess | 512 | 863 | 2,136 | — | 342 | — | 126 | 368 | 298 |
| e2e_imagenet_workflow | 512 | 723 | 1,986 | — | 330 | — | 122 | 615 | 291 |

