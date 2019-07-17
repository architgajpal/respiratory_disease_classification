import numpy as np
import pandas as pd
import librosa
import librosa.display
import glob as gb
import matplotlib.pyplot as plt
import gc

sound_dir_loc=np.array(gb.glob("audio_and_txt_files/*.wav"))

def build_spectogram(file_path,sound_file_name):
    plt.interactive(False)
    file_audio_series,sr = librosa.load(file_path,sr=None)    
    spec_image = plt.figure(figsize=[0.72,0.72])
    ax = spec_image.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    spectogram = librosa.feature.melspectrogram(y=file_audio_series, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
    image_name  = 'spec_images/' + sound_file_name + '.jpg'
    plt.savefig(image_name, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    spec_image.clf()
    plt.close(spec_image)
    plt.close('all')
    del file_path,sound_file_name,file_audio_series,sr,spec_image,ax,spectogram


index=0
for file in sound_dir_loc[index:]:
    sfile_name = file.split('\\')[1].split('.')[0]
    build_spectogram(file,sfile_name)
    print('Index:',index)
    
gc.collect()
    
# =============================================================================
# index=0
# for file in sound_dir_loc[index:index+50]:
#     sfile_name = sound_dir_loc[index].split('\\')[1].split('.')[0]
#     build_spectogram(sound_dir_loc[index],sfile_name)
#     
# index=0
# for file in sound_dir_loc[index:index+50]:
#     sfile_name = sound_dir_loc[index].split('\\')[1].split('.')[0]
#     build_spectogram(sound_dir_loc[index],sfile_name)
#     
# index=0
# for file in sound_dir_loc[index:index+50]:
#     sfile_name = sound_dir_loc[index].split('\\')[1].split('.')[0]
#     build_spectogram(sound_dir_loc[index],sfile_name)
# =============================================================================
    
patient_data=pd.read_csv('patient_diagnosis.csv',dtype=str)
spectograms_dir_loc=np.array(gb.glob("spec_images-Copy/*.jpg"))
patient_data_all = patient_data[0:0]

index = 0
for image in spectograms_dir_loc[index:]:
    image_name = image.split('\\')[1]
    patient_id = image_name.split('_')[0]
    patient_condition = patient_data.loc[(patient_data['ID'] == patient_id)].iloc[0]['CLASS']
    # print(image_name,patient_id,patient_condition,sep=' ',end='\n')
    patient_data_all.loc[len(patient_data_all)] = [image_name, patient_condition]
    
del image_name,patient_id,patient_condition,index,image

from sklearn.model_selection import train_test_split
trainset_df, testset_df = train_test_split(patient_data_all, test_size=0.2)

from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0.25)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_dataframe(
    dataframe=trainset_df,
    directory="spec_images/",
    x_col="ID",
    y_col="CLASS",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

validation_set = train_datagen.flow_from_dataframe(
    dataframe=trainset_df,
    directory="spec_images/",
    x_col="ID",
    y_col="CLASS",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

test_set = test_datagen.flow_from_dataframe(
    dataframe=testset_df,
    directory="spec_images/",
    x_col="ID",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(64,64))

from keras.models import Sequential
from keras.layers import Convolution2D, Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout,BatchNormalization


classifier = Sequential()
classifier.add(Conv2D(32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 7, activation = 'softmax'))

classifier.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])


step_train_size=training_set.n//training_set.batch_size
step_valid_size=validation_set.n//validation_set.batch_size

classifier.fit_generator(generator=training_set,
                    steps_per_epoch=step_train_size,
                    validation_data=validation_set,
                    validation_steps=step_valid_size,
                    epochs=50)

classifier.evaluate_generator(generator=validation_set, steps=step_valid_size)

step_test_size = test_set.n//test_set.batch_size
predicted_conditions = classifier.predict_generator(test_set,steps=step_test_size, verbose=1)

predicted_class_indices=np.argmax(predicted_conditions,axis=1)

labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(predictions[0:10])
print(testset_df.head(10))












