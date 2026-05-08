#!/usr/bin/env python3
"""Disabled supplement script for the public release.

The original supplement flow depends on the prompt-selection JSON, which is not
included in this release.
"""


def main():
    raise RuntimeError(
        "supplement_laws is not included in this release because it depends on "
        "the omitted prompt-selection JSON."
    )


if __name__ == "__main__":
    main()
