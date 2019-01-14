import logging as lg
import numpy as np
import rawpy as rp

logger = lg.getLogger('deluminate')


class Deluminator:
    """Many class for image processing."""

    def __init__(self, demosaic_parameters: dict = None):
        """Class constructor.

        Args:
            demosaic_parameters: Supply demosaic parameters for rawpy. If not supplied,
                uses setting that should work fine most of the time.
        """
        if not demosaic_parameters:
            demosaic_parameters = {
                'demosaic_algorithm': rp.DemosaicAlgorithm.AHD,
                'median_filter_passes': 3,
                'use_camera_wb': False,
                'no_auto_bright': True,
                'gamma': (1, 1),
                'output_bps': 16
            }
        self.demosaic_parameters = demosaic_parameters
        self.light_frames = []
        self.dark_frames = []
        self.dark_reference = None

    def load_light_frames(self, files):
        """Load light frame files.

        Args:
            files: Files with light frame raw images.
        """
        self.light_frames += self.load_raw_files(files)
        logger.info('Light frames loaded.')

    def load_dark_frames(self, files):
        """Load dark frame files and calculate reference.

        Args:
            files: Files with dark frame raw images.
        """
        self.dark_frames += self.load_raw_files(files)
        logger.info('Dark frames loaded.')
        self.get_dark_reference()

    def get_dark_reference(self):
        """Calculate dark reference."""
        self.dark_reference = np.mean(
            [image for image in self.dark_frames], 0).astype(np.uint16)
        logger.info('Dark reference calculated.')

    def load_raw_files(self, files):
        """Load a list of raw_files.

        Args:
            files: Files with raw images.

        Returns:
            A list of RGB images.
        """
        images = []

        for file in files:
            images.append(rp.imread(file).postprocess(**self.demosaic_parameters))

        return images
