import logging as lg
import matplotlib.colors as co
import matplotlib.pyplot as pp
import numpy as np
import os
import png
import rawpy as rp

logger = lg.getLogger('deluminate')


def process_directory_raw(raw_file_extension: str, degree: int,
                          path: str = None, demosaic_parameters: dict = None,
                          verbose: bool = False):
    """Process all raw files in a directory, no dark frame compensation.

    Args:
        raw_file_extension: Extension of the raw files to process.
        degree: Degree of the polynomial to fit.
        path: Path to the directory (defaults to '.').
        demosaic_parameters: Demosaic parameters for rawpy. If not supplied, uses setting that
            should work fine most of the time.
        verbose: Log progress to command line.
    """
    if not path:
        path = '.'
    if verbose:
        lg.basicConfig(level=lg.INFO)

    for file in os.listdir(path):
        if file.endswith(raw_file_extension):
            deluminator = Deluminator(demosaic_parameters)
            deluminator.load_light_frames_raw([file])
            deluminator.deluminate(degree)
            deluminator.export_deluminated()


class Deluminator:
    """Many class for image processing."""

    def __init__(self, demosaic_parameters: dict = None, mode: str = 'HSV'):
        """Class constructor.

        Args:
            demosaic_parameters: Supply demosaic parameters for rawpy. If not supplied,
                uses setting that should work fine most of the time.
            mode: 'HSV' (default) deluminate using the value channel of a HSV representation;
                'RGB' deluminate every color channel separately.
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
        self.mode = mode
        self.light_frames = []
        self.dark_frames = []
        self.dark_reference = None
        self.deluminated_frames = []

    def load_light_frames_raw(self, files):
        """Load light raw frame files.

        Args:
            files: Files with light frame raw images.
        """
        self.light_frames += self.load_raw_files(files)
        logger.info('Light frames loaded.')

    def load_dark_frames_raw(self, files):
        """Load dark frame raw files and calculate reference.

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

            if self.mode == 'RGB':
                new_image = []
                for ii in range(3):
                    channel = image[:, :, ii]
                    if self.dark_reference is not None:
                        channel -= self.dark_reference[:, :, ii]
                    poly_params = np.linalg.lstsq(fit_matrix, channel.flatten())
                    channel -= fit_matrix.dot(poly_params[0]).reshape(channel.shape)
                    new_image.append(np.clip(channel, 0, 2 ** 16 - 1).astype(np.uint16))
                self.deluminated_frames.append(np.moveaxis(np.array(new_image), 0, -1))
            elif self.mode == 'HSV':
                if self.dark_reference is not None:
                    image_hsv = co.rgb_to_hsv(np.clip(image - self.dark_reference, 0, np.inf))
                else:
                    image_hsv = co.rgb_to_hsv(image)
                poly_params = np.linalg.lstsq(fit_matrix, image_hsv[:, :, 2].flatten())
                image_hsv[:, :, 2] -= \
                    fit_matrix.dot(poly_params[0]).reshape(image_hsv[:, :, 2].shape)
                self.deluminated_frames.append(co.hsv_to_rgb(np.clip(image_hsv, 0, np.inf)))
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
            images.append(rp.imread(file).postprocess(**self.demosaic_parameters).astype(float))

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
