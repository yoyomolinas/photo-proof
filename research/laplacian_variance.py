"""
MIT License

Copyright (c) 2020 Yoel Molinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from absl import app
from absl import flags
from absl import logging

import os
import random
import time

import cv2
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image

from utils import DirectoryTree

"""
The variance of the laplacian kernel is an idicator of the 
resolution in an image. To montior the quality of pictures
taken, we would want to take a note of their laplacioan variancedatetime A combination of a date and a time. Attributes: ()

This script visualizes laplacian variances of different images.
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'the directory for the data')
flags.mark_flag_as_required("data_dir")


def main(argv):
    dt = DirectoryTree(FLAGS.data_dir)

    image_paths = dt.get_image_file_paths()
    image_paths = list(filter(lambda p: "actual" not in p, image_paths))
    random.shuffle(image_paths)

    # Visualize
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    fig.suptitle("Laplacian variances of images - an indicator of resolution")
    for i in range(16):
        img = cv2.imread(image_paths[i])
        laplacian_variance = cv2.Laplacian(img, cv2.CV_64F).var()
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title("Laplacian Variance : %.2f" % laplacian_variance)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.show()


if __name__ == "__main__":
    app.run(main)
