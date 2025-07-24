# msiconvert/convert.py
import logging
import traceback
import warnings
from pathlib import Path

from .core.base_reader import BaseMSIReader
from .core.registry import detect_format, get_converter_class, get_reader_class
from .metadata.types import EssentialMetadata

warnings.filterwarnings(
    "ignore",
    message=r"Accession IMS:1000046.*",  # or just "ignore" all UserWarning from that module
    category=UserWarning,
    module=r"pyimzml.ontology.ontology",
)

# warnings.filterwarnings(
#     "ignore",
#     category=CryptographyDeprecationWarning
# )


def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: float = None,
    handle_3d: bool = False,
    **kwargs,
) -> bool:
    """Convert MSI data to the specified format with enhanced error handling and automatic pixel size detection."""

    if not input_path or not isinstance(input_path, (str, Path)):
        logging.error("Input path must be a valid string or Path object")
        return False

    if not output_path or not isinstance(output_path, (str, Path)):
        logging.error("Output path must be a valid string or Path object")
        return False

    if not isinstance(format_type, str) or not format_type.strip():
        logging.error("Format type must be a non-empty string")
        return False

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        logging.error("Dataset ID must be a non-empty string")
        return False

    if pixel_size_um is not None and (
        not isinstance(pixel_size_um, (int, float)) or pixel_size_um <= 0
    ):
        logging.error("Pixel size must be a positive number")
        return False

    if not isinstance(handle_3d, bool):
        logging.error("handle_3d must be a boolean value")
        return False

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    logging.info(f"Processing input file: {input_path}")

    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False

    if output_path.exists():
        logging.error(f"Destination {output_path} already exists.")
        return False

    try:
        # Create reader
        input_format = detect_format(input_path)
        logging.info(f"Detected format: {input_format}")
        reader_class = get_reader_class(input_format)
        logging.info(f"Using reader: {reader_class.__name__}")
        reader = reader_class(input_path)
        should_close_reader = True

        # Handle automatic pixel size detection if not provided
        final_pixel_size = pixel_size_um
        pixel_size_detection_info = None

        if pixel_size_um is None:
            logging.info("Attempting automatic pixel size detection...")
            try:
                essential_metadata = reader.get_essential_metadata()

                if essential_metadata.pixel_size is not None:
                    final_pixel_size = essential_metadata.pixel_size[0]  # Use X size
                    logging.info(f"✓ Detected pixel size: {final_pixel_size:.1f} µm")

                    pixel_size_detection_info = {
                        "method": "automatic",
                        "detected_x_um": float(essential_metadata.pixel_size[0]),
                        "detected_y_um": float(essential_metadata.pixel_size[1]),
                        "source_format": input_format,
                        "detection_successful": True,
                        "note": "Pixel size automatically detected from source metadata",
                    }
                else:
                    logging.error("✗ Pixel size not found in metadata")
                    logging.error("Use --pixel-size parameter (e.g., --pixel-size 25)")
                    return False

            except Exception as e:
                logging.error(f"✗ Failed to extract metadata: {e}")
                logging.error("Use --pixel-size parameter (e.g., --pixel-size 25)")
                return False
        else:
            # Manual pixel size was provided
            pixel_size_detection_info = {
                "method": "manual",
                "source_format": input_format,
                "detection_successful": False,
                "note": "Pixel size manually specified via --pixel-size parameter",
            }

        # Create converter
        converter_class = get_converter_class(format_type.lower())
        logging.info(f"Using converter: {converter_class.__name__}")
        converter = converter_class(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=handle_3d,
            pixel_size_detection_info=pixel_size_detection_info,
            **kwargs,
        )

        # Run conversion
        logging.info("Starting conversion...")
        result = converter.convert()
        logging.info(f"Conversion {'completed successfully' if result else 'failed'}")

        # Only close reader if we created it
        if should_close_reader and hasattr(reader, "close"):
            reader.close()

        return result
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        # Log detailed traceback for debugging
        logging.error(f"Detailed traceback:\n{traceback.format_exc()}")

        # Ensure cleanup on exception if we created the reader
        if (
            "should_close_reader" in locals()
            and should_close_reader
            and "reader" in locals()
            and hasattr(reader, "close")
        ):
            reader.close()

        return False
