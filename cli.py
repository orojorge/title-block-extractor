"""
Command-line interface for the title block extraction pipeline.

Usage:
    python cli.py drawing.pdf
    python cli.py drawing.pdf --log-level DEBUG
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extractor import TitleBlockExtractor
from preprocessing import prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract title block metadata from an architectural PDF drawing.")
    parser.add_argument("pdf_path", help="path to the input PDF file")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(args.log_level)

    try:
        crop, thumbnail = prepare_inputs(args.pdf_path)
        extractor = TitleBlockExtractor()
        result = extractor.extract(crop=crop, thumbnail=thumbnail)
        result["path"] = str(Path(args.pdf_path).resolve())
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        print(output_json)

    except Exception as e:
        logging.getLogger(__name__).error("%s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
