# PrepPhoto2Instagram
Preparing photo to be uploaded to Instagram
Python code to prepare photo for uploading to Instagram. 
The code takes the original image and creates multiple 2048x2048 images:
1. The complete image downsampled to 2048 width/height (depending on original dimensions) with a blurred image as background
2. A detailed image of the left half on the original photo based on its height with a white background
3. A detailed image of the right half on the original photo based on its height with a white background


# Prequisties
* numpy
* opencv
* matplotlib
* pathlib
* argsparse
