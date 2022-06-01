# Collection View
A script to create an organized tableau from an image collection using unsupervise learning

## Usage 
Just pass a folder with the collection of images and it will generate one image with all thumbnails spatially organized.

Example with ~12000 images (resized)
![1200 images](/images/12000.jpg)

Example with ~200 images  (resized)
![1200 images](/images/200.jpg)


## Install

1. Clone this repository.
2. Install Conda https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
3. Prepare the environment using:  ```conda env create -f collection_view.yml```
4. Activate the environment with: ```conda activate collection_view```
5. Run the python script and pass a directory (subfolders are ok): ```python collection_view.py my_dir```
6. The image will be created one directory above the image directory.

## Options

```-h, --help```
  show this help message and exit
  
```--width WIDTH```
  width of final image
  
```--height HEIGHT```
  height of final image
  
```-t THUMBWIDTH, --thumbwidth THUMBWIDTH ```
  width of each thumbnail in the final image
  
```-m {vgg16,vgg19,resnet50}, --model {vgg16,vgg19,resnet50} ```
  choice of feature categorization model
  
```-i SAVEIMAGE, --saveimage SAVEIMAGE```
  name of image to be saved
  
```-f SAVEFEATURES, --savefeatures SAVEFEATURES```
  name of file to store the extracted features
  
```-rp RANDOMSTATE_PCA, --randomstate_pca RANDOMSTATE_PCA```
  PCA randomstate integer. repeat to obtain the same results
  
```-rt RANDOMSTATE_TSNE, --randomstate_tsne RANDOMSTATE_TSNE```
  TSNE randomstate integer. repeat to obtain the same results

## To Do

* Load features saved in the pkl file
* Clustering
* Clustering with voronoi borders
* Clustering with size weight
* Mosaic organization instead of scatter

