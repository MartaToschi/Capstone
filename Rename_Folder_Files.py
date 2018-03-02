import os

path = 'C:/Users/user/Desktop/Data Incubator/Capstone Project/Py Files/scene_film_famosi'
files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'a' + str(218 + i) +'.jpg'))
    i = i + 1

files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(218 + i) +'.jpg'))
    i = i + 1