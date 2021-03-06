# Usage
 ```
 python qt_multi_annotator.py FOLDER [classname1,classname2,...]
 ```

 * paint the markers in the image: Left mouse button draws continuously, right button draws straight line from last right button marked point
 * when mentally ready toggle the Segmentation check box (also Ctrl+T)
 * please note, changing the currently viewed image without manually saving, will revert to the last saved state, without any warning, ups

The tool will list all .png files in the given FOLDER. It is save to rename the classes in different executions of the programm, but not to change their order. The classnames of the first execution will be saved and loaded in the next executions.
The view size is currently fixed to 800x1024px2, images can be of different sizes but will be scaled. Marker images and masks are saved not in the orignal image size but in the view size. Take care, when processing further ;)

# Prerequesites
 * Python 2
 * Qt4
 * Matplotlib
 * Opencv2/3

# WIP

* This is somewhat stable but still WIP.
* Parameters are already extracted for custom image sizes, but should be fit dynamically determinable by the user.

# Demo
The below images give an impression about the workflow. First the markers are postioned/scribbled/drawn, then the regions are extended to a pixel-wise label mask.

Marker example:
![marker_img]

Label example:
![label_img]



[marker_img]: https://github.com/fkraemer/qt_multi_annotator/blob/demo/img/qt_multi_marker.png "Marker GUI Demo"
[label_img]:  https://github.com/fkraemer/qt_multi_annotator/blob/demo/img/qt_multi_mask.png "Label GUI Demo"
