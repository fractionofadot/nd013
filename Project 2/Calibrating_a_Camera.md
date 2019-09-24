# Calibrating a Camera

To start, attach a chessboard pattern to the wall and take photos of it from many different angles.

The shape of the image, which is passed into the calibrateCamera function, is just the height and width of the image. One way to retrieve these values is by retrieving them from the grayscale image shape array gray.shape[::-1]. This returns the image width and height in pixel values like (1280, 960).

