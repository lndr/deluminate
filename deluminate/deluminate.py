import rawpy as rp


class Deluminator:
    """Many class for image processing."""

    def __init__(self):
        self.light_frames_raw = []
        self.dark_frames_raw = []

    def load_light_frames(self, files):
        """Load light frame files.

        Args:
            files: Files with light frame raw images.
        """
        self.light_frames_raw.append(*self.load_raw_files(files))

    def load_dark_frames(self, files):
        """Load dark frame files.

        Args:
            files: Files with dark frame raw images.
        """
        self.dark_frames_raw.append(*self.load_raw_files(files))

    @staticmethod
    def load_raw_files(files):
        """Load a list of raw_files.

        Args:
            files: Files with raw images.
        """
        images = []

        for file in files:
            images.append(rp.imread(file))

        return images
