import os
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin



ext_path = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
model_xml = "emotions-recognition-retail-0003.xml"
model_bin = "emotions-recognition-retail-0003.bin"
emotions = ["neutral", "happy", "sad", "surprise", "anger"]

emotions_data = {}
parent_dir = "data"

# Creating the emotions_data dictionary
for emotion in emotions:
    for count, filename in enumerate(os.listdir(parent_dir+emotion)):
        if count > 25:
            break
        emotions_data[filename] = emotion


# image_paths = {"jerry.jpg" : "happy", "neutral.png" : "neutral"}
images = []

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

for i in emotions_data:
    img = cv2.imread(parent_dir + emotions_data[i] + "/" + i, cv2.IMREAD_UNCHANGED)
    print(parent_dir + emotions_data[i] + "/"+ i)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_CUBIC)
    if(img.shape == (64,64)):
        img = [img, img , img]
    img = np.reshape(img, (1,3,64,64))
    images.append((i, emotions_data[i], img))

#
#
#
# img = cv2.imread('jerry.jpg', cv2.IMREAD_UNCHANGED)
#
# cv2.imwrite("reshaped_jerry.jpg", img)
#
# img1 = cv2.imread("neutral.png", cv2.IMREAD_UNCHANGED)
# img1 = cv2.resize(img1, (64,64), interpolation = cv2.INTER_CUBIC)
# cv2.imwrite("reshaped_neutral.png", img1)
#
# img = np.reshape(img, (1,3,64,64))
# img1 = np.reshape(img1, (1,3,64,64))


device = "CPU"
plugin = IEPlugin(device = device)
plugin.add_cpu_extension(ext_path)

network = IENetwork(model = model_xml, weights = model_bin)

exec_network = plugin.load(network = network, num_requests = 2)

input_blob = next(iter(network.inputs))
output_blob = next(iter(network.outputs))

batch_size = 2
channels = 3
height = 64
width = 64
batch_size, channels, height, width = network.inputs[input_blob].shape

# results = [[image[0], image[1], [(exec_network.infer({'data' : image[2]})["prob_emotion"][0][x][0][0]) for x in range(len(emotions))], 0 , 0 , True] for image in images]
# for i in results:
#     i[3] = max(i[2])
#     i[4] = i[2][emotions.index(i[1])]
#     i[2] = emotions[i[2].index(max(i[2]))]
#
#     if(i[1] != i[2]):
#         i[5] = False

results = [[image[0], [(exec_network.infer({'data' : image[2]})["prob_emotion"][0][x][0][0]) for x in range(len(emotions))], 0 , image[1] , 0 , True] for image in images]
for i in results:
    i[2] = max(i[1])
    i[4] = i[1][emotions.index(i[3])]
    i[1] = emotions[i[1].index(max(i[1]))]

    if(i[1] != i[3]):
        i[5] = False

print(results)

import csv

with open("output.csv", "w", newline = "") as file:
    writer = csv.writer(file)
    writer.writerows(results)
