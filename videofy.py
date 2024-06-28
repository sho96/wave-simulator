import cv2
import numpy as np
import os
from tqdm import tqdm

source_folder = f"{input('source folder path: ')}"

images = [os.path.join(source_folder, f"{i + 1}.png") for i in range(len(os.listdir(source_folder)))]
fps = 30
resolution = cv2.imread(images[0]).shape[:2][::-1]

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

video = cv2.VideoWriter(input("output name: "), fourcc, fps, resolution)

for i, image in tqdm(enumerate(images), total=len(images)):
    video.write(cv2.imread(image))

video.release()
