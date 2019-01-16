import logging as lg
import matplotlib.pyplot as pp
import numpy as np
import os
import png
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
                'no_auto_bright': True,
                'gamma': (1, 1),
                'output_bps': 16
            }
        self.demosaic_parameters = demosaic_parameters
        self.light_frames = []
        self.dark_frames = []
        self.dark_reference = None
        self.deluminated_frames = []

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
            [image for image in self.dark_frames], 0)
        logger.info('Dark reference calculated.')

    def deluminate(self, degree: int = 2):
        """Remove background brightness by subtracting a polynomial of the supplied degree from
        image data.

        Args:
            degree: Degree of the polynomial to subtract.
        """
        x_grid, y_grid = np.meshgrid(range(self.light_frames[0].shape[0]),
                                     range(self.light_frames[0].shape[1]))
        fit_matrix = []
        for ii in range(degree + 1):
            for jj in range(degree + 1):
                fit_matrix.append(x_grid.flatten() ** ii * y_grid.flatten() ** jj)
        fit_matrix = np.array(fit_matrix).T
        logger.info('Matrix for delumination created.')

        for image in self.light_frames:
            new_image = []
            for ii in range(3):
                channel = image[:, :, ii].astype(float)
                if self.dark_reference is not None:
                    channel -= self.dark_reference[:, :, ii]
                poly_params = np.linalg.lstsq(fit_matrix, channel.flatten())
                channel -= fit_matrix.dot(poly_params[0]).reshape(channel.shape)
                new_image.append(np.clip(channel, 0, 2 ** 16 - 1).astype(np.uint16))
            self.deluminated_frames.append(np.moveaxis(np.array(new_image), 0, -1))
            logger.info('Image deluminated.')

    def export_deluminated(self):
        """Export the deluminated images as 16 bit pngs."""
        for image in self.deluminated_frames:
            ii = 0
            while os.path.exists('deluminated{:04d}.png'.format(ii)):
                ii += 1

            with open('deluminated{:04d}.png'.format(ii), 'wb') as file:
                writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=16)
                # Convert to  list of lists expected by the png writer.
                image_list = image.reshape(-1, image.shape[1] * image.shape[2]).tolist()
                writer.write(file, image_list)
            logger.info('Image {} exported.'.format(file.name))

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

    @staticmethod
    def preview_image(image, auto_brightness: bool = False):
        """Preview image using matplotlib.

        Args:
            image: Image from list inside Deluminator.
            auto_brightness: Auto scale brightness.
        """
        pp.figure()
        if not auto_brightness:
            pp.imshow((image / 2 ** 8).astype(np.uint8))
        else:
            scale = np.max(image)
            pp.imshow((image.astype(float) / scale))
