import os
import random
import shutil

ogfolder = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\"
dbfolder = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset"

ddl = os.listdir(ogfolder)

print(ddl)
splitSize=0.85

def split_data(SRC, TRA, VAL, SS):
    files = []
    
    # Create destination folders if they don't exist
    os.makedirs(TRA, exist_ok=True)
    os.makedirs(VAL, exist_ok=True)

    for filename in os.listdir(SRC):
        file = os.path.join(SRC, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " 0 length")
    
    print(len(files))

    trainLength = int(len(files) * splitSize)
    shuffleDS = random.sample(files, len(files))

    trainingSet = shuffleDS[:trainLength]
    validatingSet = shuffleDS[trainLength:]

    for filename in trainingSet:
        f = os.path.join(SRC, filename)
        dest = os.path.join(TRA, filename)
        shutil.copy(f, dest)

    for filename in validatingSet:
        f = os.path.join(SRC, filename)
        dest = os.path.join(VAL, filename)
        shutil.copy(f, dest)

# cloudySRC  = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\cloudy\\"
# foggySRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\foggy\\"
# rainySRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\rainy\\"
# shineSRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\shine\\"
# sunriseSRC = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset\\sunrise\\"
# cloudyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train\\cloudy\\"
# cloudyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate\\cloudy\\"

# foggyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train\\foggy\\"
# foggyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate\\foggy\\"

# rainyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train\\rainy\\"
# rainyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate\\rainy\\"

# shineTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train\\shine\\"
# shineVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate\\shine\\"

# sunriseTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train\\sunrise\\"
# sunriseVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate\\sunrise\\"

cloudySRC  = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset_filtered\\cloudy\\"
foggySRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset_filtered\\foggy\\"
rainySRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset_filtered\\rainy\\"
shineSRC   = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset_filtered\\shine\\"
sunriseSRC = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\ogdataset_filtered\\sunrise\\"


cloudyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Train\\cloudy\\"
cloudyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Validate\\cloudy\\"

foggyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Train\\foggy\\"
foggyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Validate\\foggy\\"

rainyTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Train\\rainy\\"
rainyVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Validate\\rainy\\"

shineTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Train\\shine\\"
shineVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Validate\\shine\\"

sunriseTRA = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Train\\sunrise\\"
sunriseVAL = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset_filtered\\Validate\\sunrise\\"


split_data(cloudySRC, cloudyTRA, cloudyVAL, splitSize)
split_data(foggySRC, foggyTRA, foggyVAL, splitSize)
split_data(rainySRC, rainyTRA, rainyVAL, splitSize)
split_data(shineSRC, shineTRA, shineVAL, splitSize)
split_data(sunriseSRC, sunriseTRA, sunriseVAL, splitSize)





#   split_data(ogfolder,"C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Train","C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\dataset\\Validate", splitSize)