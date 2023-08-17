import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import tensorflow_datasets as tfds
import numpy
import psutil

print(tf.__version__)
labels = ['rock', 'paper', 'scissor']

def main(): 
    args = sys.argv[0:]
    print(args)
    if len(args) == 1:
        print('Example usage:')
        print('               python rps_predict.py ./checkpoint_model.hdf5 my_test_image.jpg')
    else:
        print(' start validation')
        modelfile = args[1].replace("\\","/")
        imagefile = args[2]
        model = loadModel(modelfile)
        predict(model, imagefile)

# load sequential model and print out how much memory it used
def loadModel(modelfilename: str):
    #membefore = psutil.virtual_memory()[4]
    model = tf.keras.models.load_model(modelfilename)
    #memafter = psutil.virtual_memory()[4]
    #print('RAM memory % used before:', membefore)
    #print('RAM memory % used after:', memafter)
    #print('Memory used by model: ', str((memafter - membefore)/1024))
    return model

def predict(model: tf.keras.Sequential, testimage):
    print('start prediction')
    input_image = tf.keras.utils.load_img(testimage)
    input = tf.keras.utils.img_to_array(input_image)
    prediction = model.predict(numpy.array([input]))
    print(str(prediction))
    predclass = tf.argmax(prediction[0], axis=-1).numpy()
    print("Test Image: ", testimage)
    print("Prediction: ", labels[predclass])
    
# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()