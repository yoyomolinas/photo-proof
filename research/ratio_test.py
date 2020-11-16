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

import random
import numpy as np
import pandas as pd
import cv2

from matplotlib import pyplot as plt

import time
from tqdm import tqdm

from utils import encode, FeatureExtractor

"""
Matching features in two images consists of 
2 steps: (1) extract and match features using a
specified extractor and a brute force nearest neighbor
matcher, and (2) ratio test each match to see if it is 
a correct match. 

This script finds the best ratio and threhsold for a given dataset, 
and optionally visualizes the results.
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'the directory for the data')
flags.mark_flag_as_required("data_dir")
flags.DEFINE_string('feature_extractor', 'akaze',
                    'feature extactor you would like to use - either sift, orb, or akaze')
flags.DEFINE_bool('visualize', False,
                  'option to visualize results upon execution')
flags.register_validator(
    'feature_extractor',
    lambda value: value in ["orb", "sift", "akaze"],
    message='--feature_extractor should either be orb, sift or akaze.')


class SamplingOption:
    """Options to generate samples. 
    Read the description for generate_samples
    function for more info.
    """
    CORRECT = 1
    INCORRECT = 2
    RANDOM = 3


def generate_samples(matches, option, n=1000):
    """
    Generate n samples of Match objects for given option
    Option.CORRECT generates correct match samples where match.anchor.campaign == match.candidate.campaign
    Option.INCORRECT generates incorrect match samples where macth.anchor.campaign != match.candidate.campaign
    Option.RANDOM generates randomly matched pairs
    """
    res = []
    random.shuffle(matches)
    if option == SamplingOption.CORRECT:
        res = list(filter(lambda match: match.anchor.category ==
                          match.candidate.category, matches))[:n]
    elif option == SamplingOption.INCORRECT:
        res = list(filter(lambda match: match.anchor.category !=
                          match.candidate.category, matches))[:n]
    elif option == SamplingOption.RANDOM:
        res = matches[:n]
    else:
        raise ValueError(
            "Option argument should be an attribute of Option class")
    return res


"""
Question : What should be the ratio threshold? 
To find the threshold that yields best error rate, 
grid search best value for the ratio.
"""
N_RATIO = 150  # Number of ratio values to grid search
MIN_RATIO = 0.20  # Minimum ratio - determined experimentally
MAX_RATIO = 0.80  # Maximum ratio - determined experimentally


def main(argv):
    """Main function to run the app.
    """
    if FLAGS.feature_extractor == "sift":
        feature_extractor = FeatureExtractor.SIFT
    elif FLAGS.feature_extractor == "akaze":
        feature_extractor = FeatureExtractor.AKAZE
    elif FLAGS.feature_extractor == "orb":
        feature_extractor = FeatureExtractor.ORB

    # Encode anchors and candidates. Match features
    anchors, candidates, matches = encode(FLAGS.data_dir,
                                          read_images=True,
                                          compute_features=True,
                                          match_keypoints=True,
                                          feature_extractor=feature_extractor)

    # To find the threshold that yields best error rate,
    # grid search best value for the ratio.
    correct_matches = generate_samples(matches, SamplingOption.CORRECT)
    incorrect_matches = generate_samples(matches, SamplingOption.INCORRECT)
    random_matches = generate_samples(matches, SamplingOption.RANDOM)

    # The error function intends to minimize false negatives
    # 6 times more than false negatives. Our expermentation
    # shows that a factor of 6 is sufficient for our use cases.
    error_func = \
        lambda percent_false_positives, percent_false_negatives: 1 * \
        percent_false_positives + 0.16 * percent_false_negatives

    # Results are kept in this list
    results = {
        'ratio': [],
        'error': [],
        'percent_false_positives': [],
        'percent_false_negatives': [],
        'threshold': []
    }

    # Grid search
    logging.info("Finding best ratio and threshold for the %s feature extractor.." %
                 FLAGS.feature_extractor)
    for r in tqdm(np.linspace(MIN_RATIO, MAX_RATIO, N_RATIO)):
        # Ratio test for pre-determined correct matches
        for correct_match in correct_matches:
            correct_match.ratio_test(r)

        # Ratio test for pre-determined incorrect matches
        for incorrect_match in incorrect_matches:
            incorrect_match.ratio_test(r)

        # Define positive and negative samples. Each element in these lists represent
        # the number of keypoints over the given ratio in that match.
        positive_samples = np.array([len(match.good_matches)
                                     for match in correct_matches])
        negative_samples = np.array([len(match.good_matches)
                                     for match in incorrect_matches])

        # Sort samples to find threshold for each
        # positive_samples = np.sort(positive_samples)[::-1]
        # negative_samples = np.sort(negative_samples)[::-1]

        # Define variables for threshold grid search
        max_thresh = max(np.max(positive_samples),
                         np.max(negative_samples)) + 1e-6
        min_thresh = min(np.min(positive_samples),
                         np.min(negative_samples)) - 1e-6
        num_samples = len(positive_samples) + len(negative_samples)

        # Iterate through positive and negative samples
        # to find threshold that yields least error
        min_error = error_func(1, 1)
        threshold = None
        percent_false_positives = None
        percent_false_negatives = None
        for thresh in np.linspace(min_thresh, max_thresh, num_samples):
            percent_false_neg = len(
                positive_samples[positive_samples < thresh]) / len(positive_samples)
            percent_false_pos = len(
                negative_samples[negative_samples > thresh]) / len(negative_samples)
            err = error_func(percent_false_pos, percent_false_neg)
            if err < min_error:
                min_error = err
                threshold = thresh
                percent_false_positives = percent_false_pos
                percent_false_negatives = percent_false_neg

        # Append to results
        results['ratio'].append(r)
        results['error'].append(min_error)
        results['threshold'].append(threshold)
        results['percent_false_positives'].append(percent_false_positives)
        results['percent_false_negatives'].append(percent_false_negatives)

    df = pd.DataFrame(results)

    # Select ration and threshold that yield best result
    argmin_error = df["error"].argmin()
    error = df.iloc[argmin_error].error
    ratio = df.iloc[argmin_error].ratio
    threshold = df.iloc[argmin_error].threshold
    logging.info("With error %.4f, best ratio and threshold are %.3f and %.3f respectively" % (
        error, ratio, threshold))

    # Cross fold validate ratio and threshold for 15 folds.
    num_folds = 15
    folds = {'acc': [], 'recall': [], 'prec': []}

    logging.info(
        "Cross fold validating ratio and threshold for %i folds.." % num_folds)
    for _ in tqdm(range(num_folds)):
        random_matches = generate_samples(matches, SamplingOption.RANDOM)

        pred = []
        true = []
        for match in random_matches:
            match.ratio_test(ratio)
            num_kp = len(match.good_matches)
            pred.append(num_kp > threshold)
            true.append(match.anchor.category == match.candidate.category)

        pred = np.array(pred)
        true = np.array(true)

        time.sleep(.1)

        tp = np.sum(pred[pred == 1] == true[pred == 1])
        tn = np.sum(pred[pred == 0] == true[pred == 0])
        fp = np.sum(pred[pred == 1] != true[pred == 1])
        fn = np.sum(pred[pred == 0] != true[pred == 0])

        folds['acc'].append((tp + tn) / (tp+tn+fp+fn) * 100)
        folds['recall'].append((tp) / (tp+fn) * 100)
        folds['prec'].append((tp) / (tp+fp) * 100)

    logging.info("Average results in %i folds" % num_folds)
    logging.info("Accuracy : %.1f%%" % (np.mean(folds['acc'])))
    logging.info("Recall : %.1f%%" % (np.mean(folds['recall'])))
    logging.info("Precision : %.1f%%" % (np.mean(folds['prec'])))

    # Visualize positive and negative
    # matches in a random sequence
    if FLAGS.visualize:
        random_matches = generate_samples(
            matches, SamplingOption.RANDOM, n=1000)
        positive_matches = []
        negative_matches = []
        for match in random_matches:
            match.ratio_test(ratio)
            num_kp = len(match.good_matches)
            if num_kp > threshold:
                positive_matches.append(match)
            else:
                negative_matches.append(match)

        fig, axes = plt.subplots(4, 4, figsize=(50, 30))
        fig.suptitle("Anchors and candidates that are matched")
        axes = axes.flatten()
        for i, match in enumerate(positive_matches[:16]):
            img = cv2.cvtColor(match.drawMatchesKnn(), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        fig, axes = plt.subplots(4, 4, figsize=(50, 30))
        fig.suptitle("Anchors and candidates that are NOT matched")
        axes = axes.flatten()
        for i, match in enumerate(negative_matches[:16]):
            img = cv2.cvtColor(match.drawMatchesKnn(), cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.show()


if __name__ == "__main__":
    app.run(main)
