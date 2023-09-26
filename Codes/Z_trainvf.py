########################## CODE PERMETTANT L'ENTRAINEMENT DU MODELE AVEC SUPERVISION ####################

######################### Importation des modules ################
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import glob
import random
import keras
import segmentation_models_3D as sm
sm.set_framework('tf.keras')
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU

############################ Setup de WANDB ######################

os.environ['WANDB_API_KEY'] = '*'
os.environ['WANDB_USERNAME'] ='*'

########################## Definition du loader ###################

def load_img(img_dir,img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        if(image_name.split('.')[1]== 'npy'):
            image = np.load(img_dir+image_name)
            images.append(image)
    images = np.array(images)
    return(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L=len(img_list)
    
    
    #Keras à besoin d'un générateur infini
    while True:

        batch_start=0
        batch_end = batch_size

        while batch_start < L:
            limit= min(batch_end,L)

            X=load_img(img_dir, img_list[batch_start:limit])
            Y=load_img(mask_dir, mask_list[batch_start:limit])

            yield(X,Y)# documentation pour comprendre ce que le yield fait 

            batch_start += batch_size
            batch_end += batch_size

########################## Charger les chemins ############################
train_img_dir = "/home/mzough/projects/def-mafortin/mzough/Database/input_data_128_2channels/train/images/"
train_mask_dir = "/home/mzough/projects/def-mafortin/mzough/Database/input_data_128_2channels/train/masks/"
train_img_list = [file for file in os.listdir(train_img_dir) if not file.startswith('.DS_Store')] # présence d'un fichier étrange 
train_mask_list = os.listdir(train_mask_dir)



print ("taille de la liste des images doit etre 275", len(train_img_list))
print ("taille de la liste des masks doit etre 275", len(train_mask_list))


val_img_dir = "/home/mzough/projects/def-mafortin/mzough/Database/input_data_128_2channels/val/images/"
val_mask_dir = "/home/mzough/projects/def-mafortin/mzough/Database/input_data_128_2channels/val/masks/"
val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)



########################## Appel à notre Loader ###############################

batch_size = 4

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)


######################### Definition du reseau ##################################

kernel_initializer = 'he_uniform' 

def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

#Test if everything is working ok. 
model = simple_unet_model(128, 128, 128, 2, 4)#Taille de nos images 128 128 128 2 numéro de classes 4 
print("Taille input acceptée par le model", model.input_shape)
print("Taille input genere par le model",model.output_shape)

############################### Definition des pertes et metriques ##################################

# Cross Entropy Loss
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
total_loss = cross_entropy_loss


########################## On définit les metrics utilisées ###############
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

########################## On définit le learning rate initial ################
LR = 0.0001 ### 

########################## On définit l'optimisateur ###########################
optim = tf.keras.optimizers.legacy.Adam(LR)

########################### Batch size ##############################

####################################Chargement du modèle#################################
steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=2, #### ICI on a bien ajusté la taille du channel
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)


################################## Définition des Callback #####################

# Gestion du learning rate #

# rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou_score", factor=0.1, patience=20, verbose=1, min_lr=0.000001, min_delta=0.3, cooldown=20)

# Arret de l'entrainement en cas de non évolution #

# earlystopper = tf.keras.callbacks.EarlyStopping(patience=23, verbose=1, min_delta=0.07)


############################### Définissez une nouvelle fonction de callback pour stocker les valeurs de métriques################

class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        
        wandb.log({
            "epoch": epoch,
            "train_loss": logs.get('loss'),
            "val_loss": logs.get('val_loss'),
            "train_accuracy": logs.get('accuracy'),
            "val_accuracy": logs.get('val_accuracy'),
            "train_iou_score": logs.get('iou_score'),
            "val_iou_score": logs.get('val_iou_score')
        })



############################### Utilisation de wandb #############################

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="meningioma"

)

############################# On lance l'entrainement avec les callBack ##############
history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          callbacks=[MetricsCallback(), WandbMetricsLogger(log_freq='epoch'),
                      ]
          )

model.save('brats_3dvf.hdf5')

