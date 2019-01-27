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
            deluminator.deluminate_polynomial(degree)
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

    def load_light_frames_png(self, files):
        """Load light frames from png files.

        Args:
            files: Png files with light frames.
        """
        self.light_frames += self.load_png_files(files)
        logger.info('Light frames loaded.')

    def load_dark_frames_raw(self, files):
        """Load dark frame raw files and calculate reference.

        Args:
            files: Files with dark frame raw images.
        """
        self.dark_frames += self.load_raw_files(files)
        logger.info('Dark frames loaded.')
        self.get_dark_reference()

    def load_dark_frames_png(self, files):
        """Load dark frames from png files.

        Args:
            files: Png files with dark frames.
        """
        self.dark_frames += self.load_png_files(files)
        logger.info('Dark frames loaded.')
        self.get_dark_reference()

    def get_dark_reference(self):
        """Calculate dark reference."""
        self.dark_reference = np.mean(
            [image for image in self.dark_frames], 0)
        logger.info('Dark reference calculated.')

    def deluminate_polynomial(self, degree: int = 2, reduce_points: float = 1000):
        """Remove background brightness by subtracting a polynomial of the supplied degree from
        image data.

        Args:
            degree: Degree of the polynomial to subtract.
            reduce_points: Quotient by which the number of supporting point is reduced by.
                Points are chosen randomly.
        """
        x_grid, y_grid = np.meshgrid(range(self.light_frames[0].shape[0]),
                                     range(self.light_frames[0].shape[1]))
        fit_matrix = []
        for ii in range(degree + 1):
            for jj in range(degree + 1):
                fit_matrix.append(x_grid.flatten() ** ii * y_grid.flatten() ** jj)
        fit_matrix = np.array(fit_matrix).T
        logger.info('Matrix for delumination created.')

        point_mask = (np.random.rand(x_grid.size) - 1 / reduce_points) < 0

        for image in self.light_frames:

            if self.mode == 'RGB':
                new_image = []
                for ii in range(3):
                    channel = image[:, :, ii]
                    if self.dark_reference is not None:
                        channel = np.clip(channel - self.dark_reference[:, :, ii])
                    less_med = (channel < np.median(channel)).flatten()
                    filter_ = np.logical_and(point_mask, less_med)
                    poly_params = np.linalg.lstsq(fit_matrix[filter_],
                                                  channel.flatten()[filter_])
                    channel -= fit_matrix.dot(poly_params[0]).reshape(channel.shape)
                    new_image.append(np.clip(channel, 0, 1))
                self.deluminated_frames.append(np.moveaxis(np.array(new_image), 0, -1))
            elif self.mode == 'HSV':
                if self.dark_reference is not None:
                    image_wo_dark = np.clip(image - self.dark_reference, 0, 1)
                    image_hsv = co.rgb_to_hsv(image_wo_dark)
                else:
                    image_hsv = co.rgb_to_hsv(image)
                less_med = (image_hsv[:, :, 2] < np.median(image_hsv[:, :, 2])).flatten()
                filter_ = np.logical_and(point_mask, less_med)
                poly_params = np.linalg.lstsq(fit_matrix[filter_],
                                              image_hsv[:, :, 2].flatten()[filter_])
                image_hsv[:, :, 2] -= \
                    fit_matrix.dot(poly_params[0]).reshape(image_hsv[:, :, 2].shape)
                self.deluminated_frames.append(co.hsv_to_rgb(np.clip(image_hsv, 0, 1)))
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
                image_list = (image * 2 ** 16).reshape(-1, image.shape[1] *
                                                       image.shape[2]).tolist()
                writer.write(file, image_list)
            logger.info('Image {} exported.'.format(file.name))

    def load_raw_files(self, files):
        """Load a list of raw files.

        Args:
            files: Files with raw images.

        Returns:
            A list of RGB images.
        """
        images = []

        for file in files:
            new_image = rp.imread(file).postprocess(**self.demosaic_parameters).astype(float)
            images.append(np.clip(new_image / np.max(new_image, 0, 1)))

        return images

    @staticmethod
    def load_png_files(files):
        """Load a list of png files.

        Args:
            files: Png images.

        Returns:
            A list of RGB images.
        """
        images = []

        for file in files:
            reader = png.Reader(file)
            row_count, column_count, png_data, meta = reader.read()
            plane_count = meta['planes']
            if plane_count == 3:  # Image has no alpha channel
                new_image = np.array(list(map(np.uint16, png_data))).reshape(
                        column_count, row_count, plane_count).astype(float)
                images.append(new_image / np.max(new_image))
            elif plane_count == 4:  # iImage has alpha channel
                new_image = np.array(list(map(np.uint16, png_data))).reshape(
                            column_count, row_count, plane_count)[:, :, :3].astype(float)
                images.append(new_image / np.max(new_image))
            else:
                logger.warning('Unexpected number of planes in png image.')

        return images

    @staticmethod
    def preview_image(image: np.ndarray, auto_brightness: bool = False):
        """Preview image using matplotlib.

        Args:
            image: Image from list inside Deluminator.
            auto_brightness: Auto scale brightness.
        """
        pp.figure()
        if not auto_brightness:
            pp.imshow(image)
        else:
            pp.imshow((image / np.max(image)))
