# Contributing to Momo

Thank you for your interest in contributing to Momo! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code.

## Getting Started

1. Fork the repository: [github.com/watzon/mnemo](https://github.com/watzon/mnemo)
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mnemo.git
   cd momo
   ```
3. Build the project:
   ```bash
   cargo build
   ```
4. Run the tests:
   ```bash
   cargo test
   ```

## Development Workflow

```bash
# Run in development mode
cargo run

# Run with verbose logging
RUST_LOG=momo=debug cargo run

# Check for lint issues
cargo clippy

# Format code
cargo fmt
```

Make sure `cargo clippy` and `cargo fmt --check` pass before submitting a PR.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/watzon/mnemo/issues/new) on GitHub. Include as much detail as possible:

- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Relevant logs or error messages
- Your environment (OS, Rust version, configuration)

## Submitting Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes in small, focused commits.
3. Ensure all tests pass: `cargo test`
4. Ensure no lint warnings: `cargo clippy`
5. Ensure code is formatted: `cargo fmt`
6. Push your branch and open a PR against `main`.

Please include a clear description of what your PR does and why.

## Questions?

Feel free to [open an issue](https://github.com/watzon/mnemo/issues/new) for any questions about contributing.
