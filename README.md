# Photo Proof
The Photo Proof project matches images with photographs taken in the wild. You would use it to find photographs that contain a particular image.  A major use case is for the outdoor advertising industry where brands demand proof that their campaigns are rolling on the billboards they've paid for. The Photo Proof project is able to match patterns among smartphone shot photographs around the city and advertisement creatives.

The Photo Proof project aims to free up substantial amounts of resources in terms of labeling and matching photographs and help workforces direct their effort towards more intelligent tasks where their impact can be profound for business.

The Photo Proof project also enables grading photographs on quality, which in turn promotes better photography and higher pattern detection accuracy.

## Approach
Defining images in terms of various features has long been a research topic of interest to many scientists. The latest developments on deep neural networks and computer vision has shown that deep learning is very powerful in understanding visual contexts. However, it is expensive to acquire data to train the deep networks that specifically fits one's use case. Thus we resort to traditional feature extraction methods that have performed well for many computer vision tasks. These methods consist of but are not limited to SIFT (Scale Invariant Feature Transform), ORB (Oriented Fast Rotated Brief), and AKAZE (Accelerated KAZE). 

Our intention is to compare features between images and hopefully find a match between an anchor image and a photograph (which is denoted as a candidate image throughput the project). 
> An anchor can be any image. A positive match is found if the candidate contains the anchor in any orientation or scale.

*Example of an anchor image* 
![Anchor](https://github.com/yoyomolinas/photo-proof/blob/main/assets/mercedes_anchor.png?raw=true )

*Example of a candidate photograph shot with a smartphone camera* 
![Low Quality](https://github.com/yoyomolinas/photo-proof/blob/main/assets/mercedes_proof.jpg?raw=true )

## Method

**Steps for Feature Matching** <br/>
The essential steps for declaring positive or negative matches between an anchor images and candidate photographs are outlined below.

 1. Extract features for all images
	 > Features are defined as composed of high dimensional vectors and cartesian coordinates.
 3. Match features by recognizing nearest neighbors in clusters of features 
 4. Perform ratio test to eliminate false matches
	> The ratio is a point's distance to its closest neighbor divided by its distance to the second closest neighbor. This value must be smaller than a threshold (typically around 0.5).
 5. Recognize positive and negative matches by thresholding the number of matched keypoints
	 > The number of matched keypoints is a good indicator of whether an anchor image is in a candidate photograph. 

To empower experimentation I have prepared `ratio_test.py`, a script that spits out results for a given feature extractor. Currently, it supports the three feature extractors listed above. 

> The results are defined in terms of error, accuracy, precision, and recall and would be different for every dataset. 

*Example of a matched anchor and candidate photograph*
![Low Quality](https://github.com/yoyomolinas/photo-proof/blob/main/assets/match.png?raw=true )

**The Quality Indicator** <br/>
The higher the quality of the candidate photograph is the more likely it will find its pair anchor image. 

To quantify the quality of an image, I've decided to go with the variance of the Laplacian of the image. Higher Laplacian variance indicates better quality. 

> The Laplacian of an image is a cool way of saying edge detection

High Quality | Low Quality
--- | ---
![High Quality](https://github.com/yoyomolinas/photo-proof/blob/main/assets/high_resolution.png?raw=true) | ![Low Quality](https://github.com/yoyomolinas/photo-proof/blob/main/assets/low_resolution.png?raw=true )
has a Laplacian variance of **7887.5**. |has a Laplacian variance of **401.3**.



## Implementation

The Photo Proof project is developed in Python using OpenCV and many other great libraries. This repository is made public to support the open-source community building great things.

To recreate the environment use `conda`: 
```
conda create --name photo-proof --file ./requirements.txt
```
The `research` folder contains `ratio_test.py` and `laplacian_variance.py` files. Both scripts are designed to guide one through the methods outlined above.

**Example Usage**
```
python ratio_test.py --feature_extractor=sift --data_dir=./data
```
```
python laplacian_variance.py --data_dir=./data
```

Check out `research/utils.py` for the `DirectoryTree` class which could come in handy if you are working with folders with known structure. 
