import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--calculate_attr', help='specify the name of the attribut for which the vector should be calculated')

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from vae import model
from transformer import FaceTransformer

model.load_state_dict(torch.load('./data/weights.pth'))
attributes = [
    'image', 
    '5_o_Clock_Shadow', 
    'Arched_Eyebrows', 
    'Attractive', 
    'Bags_Under_Eyes', 
    'Bald', 
    'Bangs', 
    'Big_Lips', 
    'Big_Nose', 
    'Black_Hair', 
    'Blond_Hair', 
    'Blurry', 
    'Brown_Hair', 
    'Bushy_Eyebrows', 
    'Chubby', 
    'Double_Chin', 
    'Eyeglasses', 
    'Goatee', 
    'Gray_Hair', 
    'Heavy_Makeup', 
    'High_Cheekbones', 
    'Male', 
    'Mouth_Slightly_Open', 
    'Mustache', 
    'Narrow_Eyes', 
    'No_Beard', 
    'Oval_Face', 
    'Pale_Skin', 
    'Pointy_Nose', 
    'Receding_Hairline', 
    'Rosy_Cheeks', 
    'Sideburns', 
    'Smiling', 
    'Straight_Hair', 
    'Wavy_Hair', 
    'Wearing_Earrings',
    'Wearing_Hat', 
    'Wearing_Lipstick', 
    'Wearing_Necklace', 
    'Wearing_Necktie', 
    'Young'
]
num_samples =  200

def readAttributes():
    data = []
    with open('./data/list_attr_celeba.txt') as f:
        for line in f:
            line = line.split()
            data.append([line[0]] + [i == '1' for i in line[1:]])

    return pd.DataFrame(data, columns=attributes)

df = readAttributes()

def getAttributeSplit(attribute):
    pos_images = df.loc[df[attribute]]['image']
    neg_images = df.loc[~df[attribute]]['image']
    return pos_images.sample(frac=1), neg_images.sample(frac=1)

def imagesToBatch(image_names):
    batch = []
    totensor = transforms.ToTensor()
    for image_name in image_names:
        transformer = FaceTransformer()
        transformer.detect_faces('./images/img_align_celeba/img_align_celeba/' + image_name)
        if len(transformer.faces) > 0:
            img = transformer.faces[0].face.image.astype(np.float32)
            batch.append(totensor(img))
    return torch.stack(batch, dim=0)

def encodeImages(image_names):
    vectors = []
    for i in range(0, len(image_names), num_samples):
        print(f'processed {i} of {len(image_names)} images')
        batch = imagesToBatch(image_names.iloc[i:i+num_samples])
        vector = model.get_latent_var(batch).detach().numpy().mean(axis=0)
        vectors.append(vector)
    return vectors

def dumpAttributeVectors(attribute):
    pos_images, neg_images = getAttributeSplit(attribute)

    n = 5000
    result = {
        'pos': encodeImages(pos_images.iloc[:n]),
        'neg': encodeImages(neg_images.iloc[:n])
    }

    joblib.dump(result, './expression_vectors/' + attribute + '.pkl')

def boxplot(attribute):
    result = joblib.load('./expression_vectors/' + attribute + '.pkl')
    vectors = []
    for smile, non_smile in zip(result['pos'], result['neg']):
        vectors.append(non_smile - smile)
    vectors = np.array(vectors)
    print(vectors.shape)

    X = np.arange(vectors[0].shape[0])

    plt.boxplot(vectors)
    plt.show()

def loadVector(attribute, i):
    result = joblib.load('./expression_vectors/' + attribute + '.pkl')
    return result['pos'][i] - result['neg'][i]

def summarizeVectors():
    result = {
        'Smiling': loadVector('Smiling', 5),
        'Mustache': loadVector('Mustache', 2),
        'Young': loadVector('Young', 0)
    }
    joblib.dump(result, './expression_vectors/summary.pkl')

if __name__ == "__main__":
    args = parser.parse_args()

    print(f'calculating expression vector for attribute {args.calculate_attr}')
    dumpAttributeVectors(args.calculate_attr)