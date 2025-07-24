import logging

try:
    from . import spatialdata_converter

    logging.debug("Successfully imported spatialdata_converter")
except (ImportError, NotImplementedError) as e:
    # Skip if spatialdata dependencies not available or incompatible
    logging.error(f"SpatialData converter not available due to dependency issues: {e}")
    import traceback

    logging.error(f"Full traceback: {traceback.format_exc()}")
except Exception as e:
    logging.error(f"Unexpected error importing spatialdata_converter: {e}")
    import traceback

    logging.error(f"Full traceback: {traceback.format_exc()}")
