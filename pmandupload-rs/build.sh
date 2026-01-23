#!/usr/bin/env bash
set -euo pipefail

echo "Building pmandupload-rs (release mode)..."
cargo build --release

echo ""
echo "✓ Build complete!"
echo ""
echo "Binary location: $(pwd)/target/release/pmandupload"
echo ""
echo "Usage examples:"
echo "  ./target/release/pmandupload"
echo "  ./target/release/pmandupload --root /path/to/tiffs --jobs 8"
echo "  ./target/release/pmandupload --help"
echo ""
