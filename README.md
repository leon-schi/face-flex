# Face Flex

This is an implementation of a Variational Autoencoder (VAE) for human faces. You can use it to alter the facial expressions in images. It can automatically detect faces in an image, extract them, modify them, and then place them back into the image.

Currently the VAE can only handle 128x128 input images, so the altered face will never have a higher resolution than that. Therefore the model works best when the size of the faces in the input images is a little smaller than 128x128, we recommend 100x100.

## Examples 



## How it works

The input image will be scanned for faces with a dlib face detector, and then all the found faces will be analyzed with a shape predictor (also from dlib) that aligns 68 facial landmarks onto the face. This leads to a picture like this:

![]

Afterwards the image will be aligned, i.e. we rotate it so that the eyes will form a horizontal line and we will scale it to a uniform size of 128x128 pixels around the face:

This aligned face can then be fed into the Variational Autoencoder, i.e. it will be encoded as a 500 elements vector that represents the face.

To alter the facial expression we have a set of precomputed expression vectors that correspond to certain facial attributes such as smiling, age or gender. The facial expression can be altered by adding or subtracting the suitable expression vector to the face vector encoded by the VAE. If you e.g. want to make someone smile, you can add a multiple of the smiling vector, if you want to remove a smile, you will subtract a multiple of the smiling vector.

Now, we have a modified latent vector that can be decoded by the VAE again to obtain an actual image of the modified face:

This image is now scaled back to the original size and moved to its original location in the original image. We use the previously computed locations of the 68 facial landmarks to create a mask around the actual face. We then keep only the pixels that are part of our face and morph them into the original image. This yields the final result:

## How to use

