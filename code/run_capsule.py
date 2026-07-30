"""Thin Code Ocean entry point for paired-plane decrosstalk.

All logic lives in the ``aind-ophys-decrosstalk-roi-images-library`` package;
this wrapper only parses settings (CLI / environment) and invokes ``run``.
"""

from aind_ophys_decrosstalk_roi_images_library.job import run

if __name__ == "__main__":
    run()
