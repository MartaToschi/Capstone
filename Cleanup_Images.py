#######################################################
### CLEAN UP IMAGE FOLDER REMOVING CORRUPTED IMAGES ###
#######################################################

### IMPORT ###
import os
import sys
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from PIL import Image

### SUPPORT FUNCTIONS ###
def cleanup(path, run=False):
    print("Reading from", path)

    for root, dirs, ff in os.walk(path):
        for d in dirs:
            folder_path = join(path, d)
            print ("Examining", folder_path)
            count = 0
            for _, _, f in os.walk(folder_path):
                if len(f) == 0:
                    break
                print ("N Files, no dir", len(f))
                for i in f:
                    if(i == 'Thumbs.db'):
                        os.remove(join(folder_path, i))
                        count += 1
                    else:
                        try:
                            Image.open(join(folder_path, i))
                        except:
                            count += 1
                            print ("Remove", fp)
                            if run:
                                os.remove(join(folder_path, i))
            print ("Files removed: ", count)
            print ("--------------------------------------------")

### MAIN ###
if __name__ == "__main__":
    opt = False
    try:
        opt = sys.argv[2]
    except:
        pass
    cleanup(sys.argv[1], run=opt)