from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

imgWidth = 256
imgHeight = 256

classes = ['cloudy','foggy','rainy','shine','sunrise']

model=load_model("C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\bestWeatherModel.h5")

print(model.summary())

def prepareImage(ImagePath):
    image=load_img(ImagePath, target_size=(imgHeight,imgWidth))
    imgResult= img_to_array(image)
    imgResult= np.expand_dims(imgResult, axis=0)
    imgResult = imgResult/255.0
    return imgResult

testImagesFolder="C:/Users/Shreyas/Desktop/Weather_ANN/Test"
testImagesNamesDF= pd.read_csv("C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\test.csv")
testImagesList = []

testDFList = testImagesNamesDF['Image_id'].tolist()

#print(testDFList)

for item in testDFList:
    tempName = testImagesFolder + "/" + str(item)
    testImagesList.append(tempName)

print("List of Images")
print(testImagesList)


ImagesArray = prepareImage(testImagesList[0])
for imgName in testImagesList[1: ]:
    print("Preparing Image:" + imgName)
    processedImage = prepareImage(imgName)
    ImagesArray = np.append(ImagesArray,processedImage, axis =0 )

print(ImagesArray.shape)

np.save("C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ImagesArray.npy", ImagesArray)

resultArray= model.predict(ImagesArray, batch_size=32, verbose=1)
answers = np.argmax(resultArray, axis = 1)
print("Answers: ")
print(answers)

yTrue = testImagesNamesDF['labels']
yPred = answers

num = 0
for imgName in testImagesList:
    print("Image: " + imgName + "   True Value: " + classes[yTrue[num]] + "Predictions: " + classes[yPred[num]])
    num = num + 1