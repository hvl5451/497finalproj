import os

emotions_data = {}
parent_dir = "/Users/semideum_zepodesgan01/PycharmProjects/497FinalProj/ckplus/ck/CK+48/"
emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

# Creating the emotions_data dictionary
for emotion in emotions:
    for count, filename in enumerate(os.listdir(parent_dir+emotion)):
        if count > 21:
            break
        emotions_data[filename] = emotion

for item in emotions_data.items():
    print(item)
