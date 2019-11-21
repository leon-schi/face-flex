from rasterio import features

import matplotlib.pyplot as plt

def rasterize_polygon(landmarks, height, width):
    return features.rasterize([{ "type": "Polygon",
        "coordinates": [
            landmarks
        ]
    }], out_shape=(height, width))

if __name__ == "__main__":
    poly = rasterize_polygon([[30, 10], [40, 40], [20, 40], [10, 20], [30, 10]], 200, 200)
    plt.imshow(poly, cmap='gray')
    plt.show()