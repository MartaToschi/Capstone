import os
from random import shuffle

cont = 0
path = 'C:/Users/user/Desktop/Data Incubator/Capstone Project/Dataset/Nude/Nude_Total'
files = os.listdir(path)
shuffle(files)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'a' + str(cont + i) +'.jpg'))
    i = i + 1

files = os.listdir(path)
i = 1
for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(cont + i) +'.jpg'))
    i = i + 1