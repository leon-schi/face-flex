import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', help='specify the directory where the generated images will be stored')
parser.add_argument('--plot_landmarks', help='if specified, will additionally plot the landmarks of the detected face', action='store_true')

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from modifier import Modifier
from transformer import FaceTransformer

from config import config

def save_image(image, filename):
    img = Image.fromarray(np.uint8(image * 255))
    img.save(filename)

if __name__ == "__main__":
    args = parser.parse_args()

    plot = False
    if args.out_dir == None:
        plot = True

    filename = config['image_filename']
    transformer = FaceTransformer()
    transformer.detect_faces(config['image_filename'])

    aligned_face = transformer.faces[0]
    
    if args.plot_landmarks:
        aligned_face.plot(plt)

    attribute = config['attribute']
    sample_size = config['sample_size']
    r = config['parameter_range']
    params = np.arange(r[0], r[1], r[2])

    variations = np.random.randint(0, 20, size=sample_size)
    samples = []
    for num_variation, i in enumerate(variations):
        if plot:
            fig, ax = plt.subplots(1, len(params))

        original_face = aligned_face.face.copy()
        modifier = Modifier(aligned_face.face, attribute=attribute, i=i)

        images=[]
        for n, param in enumerate(params):
            print(f'sample: {num_variation + 1}; processing image: {config["image_filename"]} for attribute: {attribute}, parameter: {param}     ', end='\r')

            modifier.addExpression(param)
            aligned_face.face = modifier.decode_face()
            img = transformer.morph_image(aligned_face)
            images.append(img)
            modifier.addExpression(-param)
            
            if plot:
                ax[n].imshow(img)
                ax[n].set_title(f'{attribute}, {param}')
        
        aligned_face.face = original_face
        samples.append(images)

    if plot or args.plot_landmarks:
        plt.show()
    
    if not plot:
        for i, sample in enumerate(samples):
            os.mkdir(os.path.join(args.out_dir, f'sample-{i}'))
            for image, param in zip(sample, params):
                save_image(
                    image,
                    os.path.join(args.out_dir, f'sample-{i}/{param}-{attribute}.png'))