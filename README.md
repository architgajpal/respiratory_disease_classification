## Classifying Respiratory Diseases with Deep Learning
Classifying Respiratory diseases with Deep Learning using respiratory sound data.

## Objective

Respiratory sounds are important indicators of respiratory health and respiratory disorders. The sound emitted when a person breathes is directly related to air movement, changes within lung tissue and the position of secretions within the lung. A wheezing sound, for example, is a common sign that a patient has an obstructive airway disease like asthma or chronic obstructive pulmonary disease (COPD).
These sounds can be recorded using digital stethoscopes and other recording techniques. This digital data opens up the possibility of using machine learning to automatically diagnose respiratory disorders like asthma, pneumonia and bronchiolitis, to name a few.

## Data
Link to dataset from Kaggle: https://www.kaggle.com/vbookshelf/respiratory-sound-database

The Respiratory Sound Database was created by two research teams in Portugal and Greece. 

It includes 920 annotated recordings of varying length - 10s to 90s. 

These recordings were taken from 126 patients. 

There are a total of 5.5 hours of recordings containing 6898 respiratory cycles - 1864 contain crackles, 886 contain wheezes and 506 contain both crackles and wheezes. 

The data includes both clean respiratory sounds as well as noisy recordings that simulate real life conditions. The patients span all age groups - children, adults and the elderly.


This Kaggle dataset includes:

920 .wav sound files
920 annotation .txt files
A text file listing the diagnosis for each patient
A text file explaining the file naming format
A text file listing 91 names (filename_differences.txt )
A text file containing demographic information for each patient
Note:
filename_differences.txt is a list of files whose names were corrected after this dataset's creators found a bug in the original file naming script. It can now be ignored.

## Sound clips processing and Images preprocessing

Various sound clips are processed using Librosa (https://librosa.github.io/librosa/) and saved as spectograms. Resized to 0.72x0.72 and scaled uniformly. Please note that these spectograms do exhibit differences in different sound sources and cases.

## CNN Architecture

Used the processed and saved spectograms as inputs to Convolutional Neural Network with the following network architecture to classify the respiratory data:

| Layer (type)             	| Output Shape        	| Param # 	|
|--------------------------	|---------------------	|---------	|
| conv2d_1 (Conv2D)         |   (None, 62, 62, 32)   |     896    |
| max_pooling2d_1 (MaxPooling2)| (None, 31, 31, 32)        0      |  
| conv2d_2 (Conv2D)          |  (None, 29, 29, 64)  |      18496  |  
| max_pooling2d_2 (MaxPooling2)| (None, 14, 14, 64)  |      0      | 
|conv2d_3 (Conv2D)        |    (None, 12, 12, 64) |       36928    | 
|max_pooling2d_3 (MaxPooling2) |(None, 6, 6, 64)  |        0       |  
|conv2d_4 (Conv2D)        |    (None, 4, 4, 32)  |        18464     |
|max_pooling2d_4 (MaxPooling2) |(None, 2, 2, 32) |         0      |   
|dropout_1 (Dropout)        |  (None, 2, 2, 32)  |        0     |    
|flatten_1 (Flatten)       |   (None, 128)   |            0   |      
|dense_1 (Dense)       |       (None, 128)    |           16512     |
|dense_2 (Dense)         |     (None, 7)        |         903     |  

##### Total params: 92,199
##### Trainable params: 92,199
##### Non-trainable params: 0

## Results

The resultant model posed validation accuracy of 88% with loss of 39%. Due to the noise in the sound clips and slight non-uniformity in the distribution of sound data over certain labels caused the error to be high, else could be reduced to significant value.

**Out[10]: [0.3975323846465663, 0.8815789505055076]**

## Citations (on the Kaggle Dataset)

Paper: Α Respiratory Sound Database for the Development of Automated Classification
Rocha BM, Filos D, Mendes L, Vogiatzis I, Perantoni E, Kaimakamis E, Natsiavas P, Oliveira A, Jácome C, Marques A, Paiva RP (2018) In Precision Medicine Powered by pHealth and Connected Health (pp. 51-55). Springer, Singapore.
https://eden.dei.uc.pt/~ruipedro/publications/Conferences/ICBHI2017a.pdf

Ref Websites
http://www.auditory.org/mhonarc/2018/msg00007.html

http://bhichallenge.med.auth.gr/
