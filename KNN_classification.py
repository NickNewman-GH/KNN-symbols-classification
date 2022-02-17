import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from skimage.measure import label, regionprops

def get_regions(image):
    labeled = label(image)
    regions = sorted(regionprops(labeled), key=lambda region: region.bbox[1])
    return regions

def regions_separation(regions):
    sentence = []
    prev_index = 0
    bbox_dists = []
    for index in range(len(regions[:-1])):
        bbox_dists.append(regions[index + 1].bbox[1] - regions[index].bbox[3])
    for index, dist in enumerate(bbox_dists):
        if dist > np.std(bbox_dists) * 1.25:
            sentence.append(regions[prev_index:index + 1])
            prev_index = index + 1
    sentence.append(regions[prev_index:])
    return sentence

def get_images_from_regions(image, sentence):
    images_sentence = []
    is_skip_img = False
    for i, word in enumerate(sentence):
        images_sentence.append([])
        for j, char_region in enumerate(word):
            if j < len(word) - 1 and char_region.bbox[3] >= word[j + 1].bbox[1]:
                coords = [min(char_region.bbox[0], word[j + 1].bbox[0]),
                min(char_region.bbox[1], word[j + 1].bbox[1]),
                max(char_region.bbox[2], word[j + 1].bbox[2]),
                max(char_region.bbox[3], word[j + 1].bbox[3])]
                img = image[coords[0]:coords[2], coords[1]:coords[3]]
                is_skip_img = True
                continue
            if is_skip_img == True:
                is_skip_img = False
            else:
                img = char_region.image
            images_sentence[i].append(img)
    return images_sentence

def image_to_text(image, knn):
    text = []
    regions = get_regions(image)
    sentence = regions_separation(regions)
    sentence = get_images_from_regions(image, sentence)
    for word in sentence:
        for char_image in word:
            symbol = extract_features(char_image.astype("uint8"))
            symbol = np.array(symbol, dtype="f4").reshape(1, 5)
            ret, results, neighbours, dist = knn.findNearest(symbol, 3)
            text.append(chr(int(ret)))
        text.append(' ')
    return ''.join(text).strip()

def extract_features(image):
    features = []
    labeled = label(image)
    region = regionprops(labeled)[0]
    filling_factor = region.area / region.bbox_area
    features.append(filling_factor)
    centroid = np.array(region.local_centroid) / np.array(region.image.shape)
    features.extend(centroid)
    features.append(region.eccentricity)
    features.append(len(region.filled_image[region.filled_image > 0]) /  region.bbox_area)
    return features

train_dir = Path("imgs/out") / "train"
train_data = defaultdict(list)

for path in sorted(train_dir.glob("*")):
    if path.is_dir():
        for img_path in path.glob("*.png"):
            gray = cv2.imread(str(img_path), 0)
            binary = gray.copy()
            binary[gray > 0] = 1
            train_data[path.name[-1]].append(binary)

features_array = []
responses = []
for symbol in tqdm(train_data):
    for img in train_data[symbol]:
        features = extract_features(img)
        features_array.append(features)
        responses.append(ord(symbol))

features_array = np.array(features_array, dtype="f4")
responses = np.array(responses)

knn = cv2.ml.KNearest_create()
knn.train(features_array, cv2.ml.ROW_SAMPLE, responses)

for i in range(6):
    gray = cv2.imread(f"imgs/out/{i}.png", 0)
    binary = gray.copy()
    binary[gray > 0] = 1
    print(image_to_text(binary, knn))