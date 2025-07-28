# msiconvert/__main__.py
import argparse
import logging
from pathlib import Path

from msiconvert.convert import convert_msi
from msiconvert.utils.data_processors import optimize_zarr_chunks
from msiconvert.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSI data to SpatialData format"
    )

    parser.add_argument("input", help="Path to input MSI file or directory")
    parser.add_argument("output", help="Path for output file")
    parser.add_argument(
        "--format",
        choices=["spatialdata"],
        default="spatialdata",
        help="Output format type: spatialdata (full SpatialData format)",
    )
    parser.add_argument(
        "--dataset-id", default="msi_dataset", help="Identifier for the dataset"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometers. If not specified, automatic detection from "
        "metadata will be attempted. Required if detection fails.",
    )
    parser.add_argument(
        "--handle-3d",
        action="store_true",
        help="Process as 3D data (default: treat as 2D slices)",
    )
    parser.add_argument(
        "--optimize-chunks",
        action="store_true",
        help="Optimize Zarr chunks after conversion",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument("--log-file", default=None, help="Path to the log file")

    args = parser.parse_args()

    # Input validation - check early to give better error messages
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("Pixel size must be positive (got: {})".format(args.pixel_size))

    if not args.dataset_id.strip():
        parser.error("Dataset ID cannot be empty")

    # Validate input path exists and is accessible
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    # Check for common input format issues
    if input_path.is_file() and input_path.suffix.lower() == ".imzml":
        ibd_path = input_path.with_suffix(".ibd")
        if not ibd_path.exists():
            parser.error(
                f"ImzML file requires corresponding .ibd file, but not found: {ibd_path}"
            )
    elif input_path.is_dir() and input_path.suffix.lower() == ".d":
        if (
            not (input_path / "analysis.tsf").exists()
            and not (input_path / "analysis.tdf").exists()
        ):
            parser.error(
                f"Bruker .d directory requires analysis.tsf or analysis.tdf file: {input_path}"
            )

    # Check output path conflicts
    output_path = Path(args.output)
    if output_path.exists():
        parser.error(f"Output path already exists: {output_path}")

    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)

    # Pixel size handling moved entirely to convert.py
    final_pixel_size = args.pixel_size  # Can be None

    # Convert MSI data
    success = convert_msi(
        args.input,
        args.output,
        format_type=args.format,
        dataset_id=args.dataset_id,
        pixel_size_um=final_pixel_size,
        handle_3d=args.handle_3d,
    )

    if success and args.optimize_chunks:
        # Optimize chunks for better performance
        if args.format == "spatialdata":
            # For SpatialData format, optimize the table's X array
            optimize_zarr_chunks(args.output, f"tables/{args.dataset_id}/X")

    if success:
        logging.info(
            f"Conversion completed successfully. Output stored at {args.output}"
        )
    else:
        logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
