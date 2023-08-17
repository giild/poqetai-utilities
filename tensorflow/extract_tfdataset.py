import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

print(tf.__version__)
print(tfds.__version__)

def main(args):
    
    if len(args) == 1:
        print('Example usage:')
        print('        python extract_tfdataset.py dataset_name output_folder')
    else:
        print(' The images will be save to sub folders by the image class')
        datasetname = args[1]
        outputdir = args[2]
        print("dataset - ", datasetname)
        print('save to - ', outputdir)
        dataset = tfds.load(datasetname, shuffle_files=False)
        if dataset != None:
            counter = 1
            dataset = dataset["test"]
            for data in dataset:
                image = data["image"]
                label = data["label"]
                filename = str(counter) #some datasets don't have names, so I'm using a counter for the name
                strlabel = str(label.numpy())
                if "id" in data:
                    filename = data["id"].numpy().decode('utf-8')
                labeldir = outputdir + "/" + strlabel
                createDir(labeldir)
                image = image.numpy()
                savefilename = labeldir + "/" + filename + ".jpg"
                tf.keras.preprocessing.image.save_img(savefilename, image)
                counter = counter + 1
            
def createDir(directory):
    if os.path.exists(directory) == False :
        os.makedirs(directory)
        print("created directory: ", directory)
    
def loadDataset(datasetname):
    dataset = tfds.load(datasetname, shuffle_files=False, with_info=True)
    return dataset

if __name__ == "__main__":
    main(sys.argv)