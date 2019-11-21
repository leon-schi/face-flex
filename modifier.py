import torch
import joblib
import numpy as np
from torchvision import transforms

from vae import model
from image import FaceImage

model.load_state_dict(torch.load('./data/weights.pth'))
totensor = transforms.ToTensor()

def loadVector(filename, i):
    result = joblib.load(filename)
    return result['pos'][i] - result['neg'][i]

smiling_vec = loadVector('./expression_vectors/Smiling.pkl', 5)
mustache_vec = loadVector('./expression_vectors/Mustache.pkl', 2)

class Modifier:
    def __init__(self, face, attribute='Smiling', i=0):
        self.face = face
        img = face.image.astype(np.float32)
        latent = model.get_latent_var(totensor(img))
        self.latent = latent.detach().numpy()[0]
        self.loadVectors(attribute, i)

    def loadVectors(self, attribute, i):
        self.vector = loadVector('./expression_vectors/' + attribute + '.pkl', i)

    def decode_face(self):
        result = model.decode(totensor(np.array([self.latent])))
        mod_face = self.face.copy()
        img = result.detach().numpy()[0]
        img = np.array([c.T for c in img]).T
        mod_face.image = img
        return mod_face

    def addExpression(self, factor):
        self.latent += factor * self.vector