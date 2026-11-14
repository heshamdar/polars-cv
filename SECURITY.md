# Security policy

## Supported versions

polars-cv is 0.x. Security fixes go into the next release only, so upgrade to
the latest version on [PyPI](https://pypi.org/project/polars-cv/).

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/heshamdar/polars-cv/security/advisories/new)
(the repository's **Security** tab, then **Report a vulnerability**).

Include the version (`polars_cv.build_info()`), a description of the issue and,
if you can, a reproduction. You should get a reply within a week. Once a fix is
released, the advisory is published with credit to you, unless you would rather
stay anonymous.

## Scope

Areas that handle untrusted input:

- **Image decoding** of bytes from a column, a file or a URL (`image_bytes`,
  `file_path` and `auto` sources, and the header-metadata functions).
- **The path sandbox** (`allowed_roots`). A read outside the configured roots
  is a vulnerability.
- **The VIEW blob protocol** (`source("blob")`). A malformed blob must be a row
  error, never undefined behaviour.
- **Cloud credentials** (`cloud_options`, `token_command`).

Dependency advisories are checked in CI with `cargo deny`. Accepted exceptions
are listed, with their reasons, in `deny.toml`.
