# Usage
 ```
 python qt_multi_annotator.py FOLDER [classname1,classname2,...]
 ```

 * paint the markers in the image: Left mouse button draws continuously, right button draws straight line from last right button marked point
 * when mentally ready toggle the Segmentation check box (also Ctrl+T)
 * please note, changing the currently viewed image without manually saving, will revert to the last saved state, without any warning, ups

The tool will list all .png files in the given FOLDER. It is save to rename the classes in different executions of the programm, but not to change their order. The classnames of the first execution will be saved and loaded in the next executions.

# Prerequesites
 * Python 2
 * Qt4
 * Matplotlib
 * Opencv2/3

# WIP

This is somewhat stable but still WIP.
