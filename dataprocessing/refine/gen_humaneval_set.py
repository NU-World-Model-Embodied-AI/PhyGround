"""Disabled human-eval prompt-set generator for the public release.

The original generator writes prompt-selection JSON artifacts that are not part
of this code release.
"""


def main(argv=None):
    raise RuntimeError(
        "gen_humaneval_set is not included in this release because it writes "
        "prompt-selection JSON artifacts that are omitted."
    )


if __name__ == "__main__":
    main()
