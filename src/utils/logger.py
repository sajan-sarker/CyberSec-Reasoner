import logging

logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO):
    """ Set up logging configuration """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Ensure this configuration takes precedence over any existing ones
    )
    logger.info("Logging is set up.")