## `calculate_detection_performance.py`
The script reads test data (ground truth) and compares them to detected data.
Namely, it takes test data (position, orientation, size, IoU) for the tagged markers and compares those to the markers detected and their data (position, orientation, size, IoU) read from the detected counterpart marker.
When running the script, it is necessary to provide the test file and the file containing the location of detected markers by the given detection algorithm (see `get_cropped.py`).

## `create_markers.py`
An older script that was used to generate artificial markers of a given shape. The outputs were directly prepared for neural network training. The script was mainly used when designing the marker shape.

## `divide_folder.py`
The script divides the dataset into train, validation, and test subsets in the given proportion. Also, it divides the tagged information into appropriate files.

## `get_cropped.py`
The script reads information about detected markers and can crop the detected marker from the original image by the detected coordinates and preset padding.

## `get_data.py`
The main part of the solution: it takes the detected marker and returns the marker position, the marker orientation, and the location of marker corners.

## `intersection_over_union.py`
The file contains a method for IoU calculation. It also can take a test file (ground truth) and an output from the neural network and calculate the performance of the NN according to the IoU value.
