"""Shared Gemini API client and argument utilities."""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.1-pro-preview"


def add_gemini_args(parser):
    """Add standard Gemini CLI arguments to an argparse parser."""
    parser.add_argument("--api_key_file", type=str,
                        default=None,
                        help="Path to a Gemini API key file. If omitted, GEMINI_API_KEY is used.")
    parser.add_argument("--vertexai", action="store_true")
    parser.add_argument("--project", type=str, default="your-gcp-project")
    parser.add_argument("--location", type=str, default="global")


def make_client(args):
    """Create a google.genai.Client from parsed CLI arguments."""
    from google import genai
    if args.vertexai:
        client = genai.Client(vertexai=True, project=args.project,
                              location=args.location)
        logger.info("Using Vertex AI (project=%s, location=%s)",
                    args.project, args.location)
    else:
        if args.api_key_file:
            api_key = Path(args.api_key_file).read_text().strip()
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or pass --api_key_file.")
        client = genai.Client(api_key=api_key)
    return client


def extract_json_array(text):
    """Extract and parse a JSON array from LLM response text."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
