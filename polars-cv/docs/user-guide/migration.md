# Migration Guides

polars-cv is 0.x, so a minor release may change the API or a result. Each
release with something to check has a page listing exactly that; the
[changelog](../changelog.md) has everything else. Upgrading across several
releases, read each page from your version up.

| Upgrading from | To | What to check |
|---|---|---|
| 0.40 | [0.41](migration-0.41.md) | `normalize_preset` replaces `method="preset"`; `LabelReduction` is `Statistic`; `Int64` image sizes |
| 0.39 | [0.40](migration-0.40.md) | Every hash algorithm packs bits MSB-first; `hash_similarity` loses `hash_bits=`; graph wire version 2 |
| 0.38 | [0.39](migration-0.39.md) | Binary ops no longer broadcast; `channel_select` keeps the channel axis; `apply_mask`'s mask scale |
| 0.37 | [0.38](migration-0.38.md) | Two corrected results; a switch for the plugin's thread pool |
| 0.36 | [0.37](migration-0.37.md) | Two metrics results |
| 0.35 | [0.36](migration-0.36.md) | One new parameter, one edge-case result |
| 0.34 | [0.35](migration-0.35.md) | Fixes and performance; two results you may have pinned |
| 0.33 | [0.34](migration-0.34.md) | Polars 2.0 |
| 0.32 | [0.33](migration-0.33.md) | Weighted bootstrap intervals |
| 0.31 | [0.32](migration-0.32.md) | The detection-metrics layer; a higher Polars floor |
| 0.30 | [0.31](migration-0.31.md) | A removed spelling; inputs now read as written |
| 0.29 | [0.30](migration-0.30.md) | Every op computes in the dtype it declares |
| 0.28 | [0.29](migration-0.29.md) | Generated builder methods: keyword-only parameters |
