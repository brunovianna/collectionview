import tensorflow.python.keras

# for loading/processing the images
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg19 import preprocess_input

# models
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.python.keras.applications.vgg19 import VGG19


# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import seaborn as sns

from PIL import Image, ImageDraw


import argparse

# create parser
parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument("directory", default= "./", type=str, nargs=1)
parser.add_argument( "--width")
parser.add_argument( "--height")
parser.add_argument("-t", "--thumbwidth", default=120)
parser.add_argument("-m", '--model', choices=['vgg16', 'vgg19', 'resnet50'], default="resnet50")
parser.add_argument("-i", '--saveimage',default="collection_vision.jpg")
parser.add_argument("-f", '--savefeatures',default="collection_vision.pkl")
parser.add_argument("-rp", '--randomstate_pca',type=int,default=None)
parser.add_argument("-rt", '--randomstate_tsne',type=int,default=None)

# parse the arguments
args = parser.parse_args()



# this list holds all the image filename
pictures = []

# creates a ScandirIterator aliased as files
for path, dirs, files in os.walk(args.directory[0]):
  # loops through each file in the directory
    for filename in files:
        fullpath = os.path.join(path, filename)
        if filename.endswith('.jpg'):
          # adds only the image files to the flowers list
            pictures.append(fullpath)

print ("\n\nFound %d pictures" %len(pictures))


if args.width is None:
    if args.height is None:
        #automate w and w
        height = width = int(np.sqrt(len(pictures)) * (args.thumbwidth ) * 1.5)
        print ("image will be "+str(height)+"x"+str(width))

    else:
        height = width = args.height
        print ("width not given, making a square of "+str(height)+" pixels")
else:
    if args.height is None:
        height = width = args.width
        print ("height not given, making a square of "+str(height)+" pixels")
    else:
        width = args.width
        height = args.height
        print ("image will be "+str(height)+"x"+str(width))

print ("Extracting features...\n\n")

modelshape = 0

if args.model == "vgg16":
    model = VGG16()
    modelshape = 4096
elif args.model == "vgg19":
    model = VGG19()
    modelshape = 4096
elif args.model == "resnet50":
    model = ResNet50(weights='imagenet')
    modelshape = 2048


model = Model(inputs = model.inputs, outputs = model.layers[-2].output)



def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    if args.model == "vgg16":
        imgx = vgg16_preprocess(reshaped_img)
    elif args.model == "vgg19":
        imgx = vgg19_preprocess(reshaped_img)
    elif args.model == "resnet50":
        imgx = resnet50_preprocess(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

data = {}
p = args.directory[0]+"/collection_data.pkl"

# lop through each image in the dataset
for index, picture in enumerate(pictures):
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(picture,model)
        data[picture] = feat
        print (index, picture)
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p,'wb') as file:
            pickle.dump(data,file)

# save the extracted features as a pickle file (optional)
with open(p,'wb') as file:
    pickle.dump(data,file)

print ("\n\nSaved feature data in file '"+args.directory[0]+"collection_data.pkl'\n\n")

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are same vectors as last layer
feat = feat.reshape(-1,modelshape)

#reduce to 100 dimensions (unless there are less than 100 pics)
components = 100
if len(data) < 100:
    components = len (data)

pca = PCA(n_components=components, random_state=args.randomstate_pca)
pca.fit(feat)
x = pca.transform(feat)

#reduce to 2 dimensions with TSNE
model_tsne = TSNE(random_state=args.randomstate_tsne)
tsne_scatter = model_tsne.fit_transform (x)

tsne_df = pd.DataFrame(data = tsne_scatter, columns = ['x', 'y'])

scale_x = 0.8 * (width + args.thumbwidth ) / (tsne_df['x'].max()-tsne_df['x'].min())
scale_y = 0.8 *  (height +args.thumbwidth ) / (tsne_df['y'].max()-tsne_df['y'].min())


middle_x = (tsne_df['x'].max()-tsne_df['x'].min()) / 2
middle_y = (tsne_df['y'].max()-tsne_df['y'].min()) / 2


scaled_x = tsne_df['x'].subtract(other=tsne_df['x'].min()).multiply(other=scale_x).add(other=(args.thumbwidth))
scaled_y = tsne_df['y'].subtract(other=tsne_df['y'].min()).multiply(other=scale_y).add(other=(args.thumbwidth))


pixel_frame = pd.DataFrame({'x': scaled_x, 'y': scaled_y})
pixel_frame['filename'] = filenames


output = Image.new('RGB',(width,height), color='white')


for index, row in pixel_frame.iterrows():


    img = Image.open(row['filename'])

    thumbheight = args.thumbwidth * (img.height / img.width)
    img = img.resize((int(args.thumbwidth),int(thumbheight)))

    output.paste (img,box=(int(row['x']-img.size[0]),int(row['y']-img.size[1])))

if args.directory[0] == "./":
    output.save(args.directory[0]+args.saveimage)
    print ("saved as "+args.directory[0]+args.saveimage)
else:
    output.save(args.directory[0]+'../'+args.saveimage)
    os.chdir(args.directory[0])
    os.chdir("..")
    print ("saved as "+os.getcwd()+'/'+args.saveimage)
