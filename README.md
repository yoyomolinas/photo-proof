# Photo Proof
The Photo Proof project matches images with photographs taken in the wild. You would use it to find photographs that contain a particular image.  A major use case is for the outdoor advertising industry where brands demand proof that their campaigns are rolling on the billboards they've rented. Photo proof is able to match patterns among photographs shot around the city from smartphone cameras and advertisement creatives.

The Photo Proof project aims to free up substantial amounts of resources in terms of labeling and matching photographs and help workforces direct their effort towards more intelligent tasks where their impact can be profound for business.

The Photo Proof project also enables grading photographs on resolution, which directly impacts the pattern detection accuracy.

## Approach
Extracting features from images has long been a research topic of interest to many scientists. The latest developments on deep neural networks and computer vision has shown that deep learning can be a very powerful for understanding visual contexts. However, it is expensive to acquire data that specifically fits one's use case. Thus we resort to traditional feature extraction methods that have performed well for many computer vision tasks. This methods consist of but are not limited to SIFT (Scale Invariant Feature Transform), ORB (Oriented Fast Rotated Brief), and AKAZE (Accelerated KAZE). 

Our intention is to compare features between images and hopefully find a match between an anchor image and a photograph which is denoted as a candidate throughput the project. 
> An anchor can be any image. A positive match is found if the candidate contains the anchor in any orientation or scale.

### Method

**Steps for Feature Matching**
The essential steps before declaring a positive or a negative match between an anchor image and a photograph in the wild are outlined below.

 1. Extract features for all images
	 > Features are defined as high dimensional vectors that correspond to cartesian coordinates.
 3. Match features by recognizing nearest neighbors in clusters of features 
 4. Perform ratio test to eliminate false matches
	> The ratio is a point's distance to its closest neighbor divided by its distance to the second closest neighbor. This value must be smaller than a threshold (typically around 0.5).
 5. Recognize positive and negative matches by thresholding the number of matched keypoints
	 > The number of matched keypoints is a good indicator of whether an anchor image is in a candidate photograph. 

To empower experimentation we have prepared `ratio_test.py`, a script that spits out results for a given feature extractor. We currently support the three feature extractors listed above. 

> The results are defined in terms of error, accuracy, precision, and recall and would be different for every dataset. 

**The Resolution Indicator**
The resolution of a photograph is a good indicator of the likelihood that it will find its pair anchor image. 

To quantify the resolution, we use the variance of the Laplacian of an image. 

