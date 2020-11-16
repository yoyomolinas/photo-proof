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
from os.path import join
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def imread(path, mode="bgr"):
    """
    Helper function to read images
    Mode could be one of 'bgr' or 'rgb'
    """
    img = cv2.imread(path)
    if mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class DirectoryTree:
    """
    A DirectoryTree class using composite pattern. 
    A directory tree has a path for itself, a parent
    directory, and child directories. The parent and
    child directories are instances of the DirectoryTree class.
    """

    def __init__(self, path=None, parent=None, depth=0):
        self.parent = parent
        self.path = path
        self.directories = {}
        self.depth = depth
        if depth == 0:
            self.name = self.path
        else:
            self.name = self.path.split('/')[-1]

        if os.path.isfile(self.path):
            raise OSError('Please specify a directory not a file!')

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            # Iterate through all directories in self.path, and add to self.directories
            for dir in os.listdir(self.path):
                if os.path.isdir(join(self.path, dir)):
                    self.add(dir)

    def add(self, *names):
        """ Add folders to current tree."""
        if not self.exists():
            raise OSError('This directory tree is no longer valid.')
        for name in names:
            if hasattr(self, name):
                raise OSError(
                    'path <%s> already exists in this file structure' % join(self.path, name))

            setattr(self, name, DirectoryTree(
                path=join(self.path, name), parent=self, depth=self.depth + 1))
            self.directories[name] = getattr(self, name)

    def print_all(self):
        """
        A debug function. Prints all directories 
        in the tree in an understandable format. 
        """
        if not self.exists():
            raise OSError('This directory tree is no longer valid.')
        cur_path = self.path.split('/')[-1]
        s = ''
        if self.depth != 0:
            s = '|'
            for i in range(self.depth):
                s += '___'

        logging.info("%s%s" % (s, cur_path))
        for name, d in self.directories.items():
            d.logging.info_all()

    def remove(self, force=False):
        """
        Delete a directory tree. 
        If the directory is not empty
        use the force option for force
        removal. 
        """
        if not self.exists():
            raise OSError('This directory tree is no longer valid.')
        if force:
            shutil.rmtree(self.path)
        else:
            os.rmdir(self.path)

        if self.parent is not None:
            delattr(self.parent, self.name)
            del self.parent.directories[self.name]

    def exists(self):
        return os.path.isdir(self.path)

    def get_file_paths(self, exts=[""]):
        """
        Find all files with given extensions. 
        This is particularly useful to query 
        sepcific files scattered in a directory
        tree of unknown structure. 
        """
        res = []
        # Add files in current directory
        for f in os.listdir(self.path):
            file_path = os.path.join(self.path, f)
            if not os.path.isfile(file_path):
                continue
            for ext in exts:
                size_ext = len(ext)
                if size_ext > len(file_path):
                    continue
                if file_path[-size_ext:] == ext:
                    res.append(file_path)

        # Add files in sub directories
        for dt in self.directories.values():
            res.extend(dt.get_file_paths(exts=exts))
        return res

    def get_image_file_paths(self):
        """
        Helper function to all images in 
        this directory tree. 
        """
        exts = ['.JPG', '.jpg', '.png', '.PNG']
        return self.get_file_paths(exts=exts)


class FeatureExtractor:
    SIFT = 1
    AKAZE = 2
    ORB = 3


class FeatureMatcher:
    BRUTE_FORCE = 1
    BRUTE_FORCE_HAMMING = 2


class Encoder:
    """
    Base class to encode anchor or candidate images.
    """
    feature_extractors = {
        FeatureExtractor.SIFT: cv2.SIFT_create(),
        FeatureExtractor.AKAZE: cv2.AKAZE_create(),
        FeatureExtractor.ORB: cv2.ORB_create()
    }

    def __init__(self, category, path, feature_extractor=FeatureExtractor.ORB):
        self.category = category
        self.path = path
        self.id = hash(self.path)
        self.kp = None
        self.des = None
        self.img = None

        assert feature_extractor in [
            FeatureExtractor.SIFT, FeatureExtractor.AKAZE, FeatureExtractor.ORB], "Feature extractor should be either be sift, orb, or akaze."
        self.feature_extractor = feature_extractor

    def read(self, mode="bgr"):
        """
        Read img in self.path to self.img 
        """
        if self.img is None:
            img = imread(self.path, mode=mode)
            np_img = np.array(img)
            self.img = np_img

    def get_image(self):
        """
        Returns self.img if not None
        """
        if not self.img:
            self.read()
        return self.img

    def extract_features(self):
        self.read()
        if (not self.kp) or (not self.des):
            kp, des = Encoder.feature_extractors[self.feature_extractor].detectAndCompute(
                self.img, None)
            self.kp = kp
            self.des = des

    def get_keypoints(self):
        if not self.kp:
            self.extract_features()
        return self.kp

    def get_descriptors(self):
        if not self.des:
            self.extract_features()
        return self.des


class Candidate(Encoder):
    """
    Class to encode candidate images. 
    """

    def __init__(self, category, path, feature_extractor=FeatureExtractor.ORB):
        super().__init__(category, path, feature_extractor)


class Anchor(Encoder):
    """
    Class to encode anchor images. 
    """

    def __init__(self, category, path, feature_extractor=FeatureExtractor.ORB):
        super().__init__(category, path, feature_extractor)


class Match:
    """
    Class that represents a match between anchors and candidates.
    """

    feature_matchers = {
        FeatureMatcher.BRUTE_FORCE: cv2.BFMatcher(),
        FeatureMatcher.BRUTE_FORCE_HAMMING: cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
    }

    def __init__(self, anchor, candidate, feature_matcher=FeatureMatcher.BRUTE_FORCE_HAMMING):
        self.anchor = anchor
        self.candidate = candidate
        self.matches = None
        self.good_matches = None
        self.feature_matcher = feature_matcher

    def match_keypoints(self):
        """Match keypoints using the nearest neightbor approach."""
        self.matches = Match.feature_matchers[self.feature_matcher].knnMatch(
            self.anchor.des, self.candidate.des, k=2)

    def ratio_test(self, r=0.5):
        """      
        Find good matches with a ratio test.
        If the ratio of two neighboring keypoints 
        (in the feature space) is below a specified
        threshold then this is considered to be a 
        good match.

        Args:
            r (float, optional): ratio between 0 and 1. Defaults to 0.5.
        """
        assert self.matches, "keypoints should be matched before doing ratio test"
        # Ratio test
        assert r > 0 and r < 1, "ratio should be between 0 and 1 but is %.2f" % r
        good = []
        for m, n in self.matches:
            if m.distance < r * n.distance:
                good.append([m])
        self.good_matches = good

    def drawMatchesKnn(self):
        """A debugging method to visualize matches.

        Returns:
            array: image with matched keypoints from reference and current
        """
        matches = self.good_matches if self.good_matches is not None else self.matches
        drawing = cv2.drawMatchesKnn(
            self.anchor.img,
            self.anchor.kp,
            self.candidate.img,
            self.candidate.kp,
            matches,
            None, flags=2)
        return drawing


def encode(
        root_path: str,
        read_images=False,
        compute_features=False,
        match_keypoints=False,
        feature_extractor=FeatureExtractor.ORB):
    """Helper function to encode images. This 
    function takes the data folder structure 
    into consideration. The data should be organized 
    in the following manner: 
        root 
        ---- [category name]
        --------------- actual (holds anchor images)
        --------------- proof (holds matching images to the anchor)

    Args: 
        root_path (str): root data path
        read_images (bool, optional): Defaults to False.
        compute_features (bool, optional): Defaults to False.
        match_keypoints (bool, optional): Defaults to False.
        feature_extractor (str, optional): . Defaults to "sift".

    All optional arguments support the felxibility of the processing pipeline. 
    These options are used to optimize calls to the encode function. 
    Ine might make several calls to encode in different stages of processing 
    the images. 

    Returns:
        tuple: A tuple of anchors, candidates, matches in that order.
    """
    # Select feature matcher based on extractor
    feature_matcher = FeatureMatcher.BRUTE_FORCE
    if feature_extractor in [FeatureExtractor.ORB, FeatureExtractor.AKAZE]:
        feature_matcher = FeatureMatcher.BRUTE_FORCE_HAMMING

    anchors, candidates = [], []
    root_dt = DirectoryTree(root_path)
    logging.info("Encoding anchors and candidates..")
    for dt in tqdm(root_dt.directories.values()):
        category = dt.name

        # Find & encode anchor
        tmp = dt.actual.get_image_file_paths()
        assert len(tmp) == 1, "There should be only single anchor image but found %i in %s" % (
            len(tmp), dt.actual.path)
        anchor_path = tmp[0]
        anchor = Anchor(category, anchor_path, feature_extractor)
        if read_images:
            anchor.read()
        if compute_features:
            anchor.extract_features()
        anchors.append(anchor)

        # Find & encode candidates
        tmp = dt.proof.get_image_file_paths()
        assert len(tmp) > 0, "There should more than zero candidates"
        candidate_paths = tmp
        for candidate_path in candidate_paths:
            candidate = Candidate(category, candidate_path, feature_extractor)
            if read_images:
                candidate.read()
            if compute_features:
                candidate.extract_features()
            candidates.append(candidate)

    # Create a list of possible matches
    # This is a anchor x candidate pairing operation
    # with O(anchors * candidates)
    matches = []
    for anchor in anchors:
        for candidate in candidates:
            matches.append(
                Match(anchor, candidate, feature_matcher=feature_matcher))

    logging.info("Matching anchors and candidates..")
    # Match keypoints in every match
    if match_keypoints:
        assert compute_features, "To do matching use compute_features option."
        for match in tqdm(matches):
            match.match_keypoints()

    return anchors, candidates, matches
