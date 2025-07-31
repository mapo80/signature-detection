#!/usr/bin/env python3
import base64
import argparse
import sys
from pathlib import Path

def decode_base64_file(input_path: Path, remove_original: bool = False) -> None:
    """
    Decode a Base64‐encoded file and write the binary image, optionally deleting the source file.

    :param input_path: Path to the .base64 file
    :param remove_original: Whether to delete the .base64 file after decoding
    """
    try:
        b64_string = input_path.read_text().strip()
        # Handle data URI header if present
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]
        image_data = base64.b64decode(b64_string)
    except Exception as e:
        print(f"Skipping {input_path}: error decoding ({e})", file=sys.stderr)
        return

    # Determine output file path by removing .base64 suffix
    output_path = input_path.with_suffix("")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_data)
        print(f"Decoded: {input_path.name} -> {output_path.name}")
    except Exception as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
        return

    if remove_original:
        try:
            input_path.unlink()
            print(f"Removed original: {input_path.name}")
        except Exception as e:
            print(f"Error removing {input_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Batch decode all .base64 files in a directory and remove originals."
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory containing .base64 files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save decoded images (defaults to input directory)"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recursively process subdirectories"
    )
    parser.add_argument(
        "--remove-original", action="store_true",
        help="Remove .base64 files after successful decoding"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.output_dir or input_dir
    remove_flag = args.remove_original

    if not input_dir.is_dir():
        sys.exit(f"Error: {input_dir} is not a directory.")

    pattern = "**/*.base64" if args.recursive else "*.base64"
    for b64_file in input_dir.glob(pattern):
        # Optionally change working dir for output-dir
        if out_dir != input_dir:
            # maintain relative path
            relative = b64_file.relative_to(input_dir)
            target = out_dir.joinpath(relative)
        else:
            target = b64_file
        # Temporarily assign to input_path for decoding
        input_path = b64_file
        # Change suffix for output
        # The decode function uses input_path for naming
        decode_base64_file(input_path, remove_flag)

if __name__ == "__main__":
    main()