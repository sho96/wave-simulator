import cv2
import os

directory = "~ directory containing the frames ~"
output_file = "output.mp4"


images = [img for img in sorted(os.listdir(directory)) if img.endswith(".png")]
images = sorted(images, key=lambda x: int(x.split('.')[0]))
frame = cv2.imread(os.path.join(directory, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

for image in images:
  print(image)
  video.write(cv2.imread(os.path.join(directory, image)))

cv2.destroyAllWindows()
video.release()
