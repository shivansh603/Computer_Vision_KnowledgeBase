![Norfair by Tryolabs logo](https://raw.githubusercontent.com/tryolabs/norfair/master/docs/img/banner.png)

## Description

Norfair is a customizable lightweight Python library for real-time multi-object tracking.Using Norfair, you can add tracking capabilities to any detector with just a few lines of code.

|                                           Tracking players with moving camera                                           |                                           Tracking 3D objects                                           |
| :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
| ![Tracking players in a soccer match](https://raw.githubusercontent.com/tryolabs/norfair/master/docs/videos/soccer.gif) | ![Tracking objects in 3D](https://raw.githubusercontent.com/tryolabs/norfair/master/docs/videos/3d.gif) |

## Features

- Any detector expressing its detections as a series of `(x, y)` coordinates can be used with Norfair. This includes detectors performing tasks such as object or keypoint detection (see [examples](#examples--demos)).

- Modular. It can easily be inserted into complex video processing pipelines to add tracking to existing projects. At the same time, it is possible to build a video inference loop from scratch using just Norfair and a detector.

- Supports moving camera, re-identification with appearance embeddings, and n-dimensional object tracking (see [Advanced features](#advanced-features)).

- Norfair provides several predefined distance functions to compare tracked objects and detections. The distance functions can also be defined by the user, enabling the implementation of different tracking strategies.

- Fast. The only thing bounding inference speed will be the detection network feeding detections to Norfair.

Norfair is built, used and maintained by [Tryolabs](https://tryolabs.com).

If the needed dependencies are already present in the system, installing the minimal version of Norfair is enough for enabling the extra features. This is particularly useful for embedded devices, where installing compiled dependencies can be difficult, but they can sometimes come preinstalled with the system.

## Documentation

[Getting started guide](https://tryolabs.github.io/norfair/latest/getting_started/).

[Official reference](https://tryolabs.github.io/norfair/latest/reference/).


### Benchmarking and profiling

1. [Kalman filter and distance function profiling](https://github.com/tryolabs/norfair/tree/master/demos/profiling) using [TRT pose estimator](https://github.com/NVIDIA-AI-IOT/trt_pose).
2. Computation of [MOT17](https://motchallenge.net/data/MOT17/) scores using [motmetrics4norfair](https://github.com/tryolabs/norfair/tree/master/demos/motmetrics4norfair).

![Norfair OpenPose Demo](https://raw.githubusercontent.com/tryolabs/norfair/master/docs/videos/openpose_skip_3_frames.gif)

## How it works

Norfair works by estimating the future position of each point based on its past positions. It then tries to match these estimated positions with newly detected points provided by the detector. For this matching to occur, Norfair can rely on any distance function. There are some predefined distances already integrated in Norfair, and the users can also define their own custom distances. Therefore, each object tracker can be made as simple or as complex as needed.

As an example we use [Detectron2](https://github.com/facebookresearch/detectron2) to get the single point detections to use with this distance function. We just use the centroids of the bounding boxes it produces around cars as our detections, and get the following results.

![Tracking cars with Norfair](https://raw.githubusercontent.com/tryolabs/norfair/master/docs/videos/traffic.gif)

On the left you can see the points we get from Detectron2, and on the right how Norfair tracks them assigning a unique identifier through time. Even a straightforward distance function like this one can work when the tracking needed is simple.

## Motivation

Trying out the latest state-of-the-art detectors normally requires running repositories that weren't intended to be easy to use. These tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric to publish on a particular research paper. This explains why they tend to not be easy to run as inference scripts, or why extracting the core model to use in another standalone script isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. It was designed to seamlessly be plugged into a complex, highly coupled code base, with minimum effort. Norfair provides a series of modular but compatible tools, which you can pick and choose to use in your project.

## Comparison to other trackers

Norfair's contribution to Python's object tracker library repertoire is its ability to work with any object detector by being able to work with a variable number of points per detection, and the ability for the user to heavily customize the tracker by creating their own distance function.

If you are looking for a tracker, here are some other projects worth noting:

- [**OpenCV**](https://opencv.org) includes several tracking solutions like [KCF Tracker](https://docs.opencv.org/3.4/d2/dff/classcv_1_1TrackerKCF.html) and [MedianFlow Tracker](https://docs.opencv.org/3.4/d7/d86/classcv_1_1TrackerMedianFlow.html) which are run by making the user select a part of the frame to track, and then letting the tracker follow that area. They tend not to be run on top of a detector and are not very robust.
- [**dlib**](http://dlib.net) includes a correlation single object tracker. You have to create your own multiple object tracker on top of it yourself if you want to track multiple objects with it.
- [**AlphaPose**](https://github.com/MVIG-SJTU/AlphaPose) just released a new version of their human pose tracker. This tracker is tightly integrated into their code base, and to the task of tracking human poses.
- [**SORT**](https://github.com/abewley/sort) and [**Deep SORT**](https://github.com/nwojke/deep_sort) are similar to this repo in that they use Kalman filters (and a deep embedding for Deep SORT), but they are hardcoded to a fixed distance function and to tracking boxes. Norfair also adds some filtering when matching tracked objects with detections, and changes the Hungarian Algorithm for its own distance minimizer. Both these repos are also released under the GPL license, which might be an issue for some individuals or companies because the source code of derivative works needs to be published.

## Benchmarks

[MOT17](https://motchallenge.net/data/MOT17/) and [MOT20](https://motchallenge.net/data/MOT17/) results obtained using [motmetrics4norfair](https://github.com/tryolabs/norfair/tree/master/demos/motmetrics4norfair) demo script on the `train` split. We used detections obtained with [ByteTrack's](https://github.com/ifzhang/ByteTrack) YOLOX object detection model.

| MOT17 Train |   IDF1 IDP IDR    | Rcll  | Prcn  |  MOTA MOTP  |
| :---------: | :---------------: | :---: | :---: | :---------: |
|  MOT17-02   | 61.3% 63.6% 59.0% | 86.8% | 93.5% | 79.9% 14.8% |
|  MOT17-04   | 93.3% 93.6% 93.0% | 98.6% | 99.3% | 97.9% 07.9% |
|  MOT17-05   | 77.8% 77.7% 77.8% | 85.9% | 85.8% | 71.2% 14.7% |
|  MOT17-09   | 65.0% 67.4% 62.9% | 90.3% | 96.8% | 86.8% 12.2% |
|  MOT17-10   | 70.2% 72.5% 68.1% | 87.3% | 93.0% | 80.1% 18.7% |
|  MOT17-11   | 80.2% 80.5% 80.0% | 93.0% | 93.6% | 86.4% 11.3% |
|  MOT17-13   | 79.0% 79.6% 78.4% | 90.6% | 92.0% | 82.4% 16.6% |
|   OVERALL   | 80.6% 81.8% 79.6% | 92.9% | 95.5% | 88.1% 11.9% |

| MOT20 Train |   IDF1 IDP IDR    | Rcll  | Prcn  |  MOTA MOTP  |
| :---------: | :---------------: | :---: | :---: | :---------: |
|  MOT20-01   | 85.9% 88.1% 83.8% | 93.4% | 98.2% | 91.5% 12.6% |
|  MOT20-02   | 72.8% 74.6% 71.0% | 93.2% | 97.9% | 91.0% 12.7% |
|  MOT20-03   | 93.0% 94.1% 92.0% | 96.1% | 98.3% | 94.4% 13.7% |
|  MOT20-05   | 87.9% 88.9% 87.0% | 96.0% | 98.1% | 94.1% 13.0% |
|   OVERALL   | 87.3% 88.4% 86.2% | 95.6% | 98.1% | 93.7% 13.2% |

