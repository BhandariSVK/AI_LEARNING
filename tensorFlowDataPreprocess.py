import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

#! HANDEL MODEL VERSION
MODEL_VERSION = 1
##############################
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
FIG_SIZE = 10
EPOCHS = 5
TARIN = 0.8 # 80% Training
VLD_TST = 0.5 # Reamining 50% Testing & VALIDATION
SUFFLE_SIZE = 10000 # Will hold max 10000 into RAM as Buffer
SEED = 12 # Produce Same Random Oder because = 12 Every time, WIthout this transorFlow Gives A New Random Oder For Randomization IN Every Run ! Its Hard To Debug
MODEL_INP_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
N_CLASSES = 3

dataset  = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\SOUVIK\\AI_LEARNING\\Downloads\\POTATO",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

print(dataset.class_names) # Folder Names are Class Names
print(len(dataset))

######################################## SHOWING IMAGE AS A FRAME ###########################################
# plt.figure(figsize=(FIG_SIZE,FIG_SIZE))
# for image_batch, lebel_batch in dataset.take(1):
#     # print(image_batch.shape)
#     # print(image_batch[0].shape)
#     for i in range(12):
#         ax = plt.subplot(3, 4, i+1)
#         plt.imshow(image_batch[i].numpy().astype("uint8"))
#         plt.title(dataset.class_names[lebel_batch[i]])
#         plt.axis("off")
# plt.show()


#! Suffling
dataset = dataset.shuffle(SUFFLE_SIZE, seed=SEED)

#! Seprating Data For Train And TEst And Validate
testCount = len(dataset) * TARIN

train_dataset = dataset.take(tf.cast(tf.floor(testCount), tf.int64))

remaining_dataset_VL = dataset.skip(tf.cast(tf.floor(testCount), tf.int64))
 
vld_cnt = len(remaining_dataset_VL) * VLD_TST 
val_dataset = remaining_dataset_VL.take(tf.cast(tf.floor(vld_cnt), tf.int64))

tst_dataset = remaining_dataset_VL.skip(tf.cast(tf.floor(vld_cnt), tf.int64))


#! PREPARE PIPELINE FOR TRANSORFLOW TRAINING
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
tst_dataset = tst_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


#! DATA RESIZE AND RESCALE
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),        #Ensures all images have the same shape, IF User Input Is Various
    layers.Rescaling(1.0/255)                      #Normalizes pixel values, if pixel=150 then it does=150 / 255 = 0.588, Because We only need Similarity not actual Image, although this helps ::::::>>>>>> Neural networks use: gradients, backpropagation, weights update, >>> small values help in make training smooth, faster convergence, reduce mathematical instability
])

#! DATA AUGUMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),   #Augments images (flip + rotate) to avoid overfitting
    layers.RandomRotation(0.2)                      #Randomly rotates images by ±20% of 360°. 0.2 × 360 = 72 degrees
])


#! Prepare Model
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=MODEL_INP_SHAPE),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax'),
])

#! Build The Model
model.build(input_shape=MODEL_INP_SHAPE)

#! Get And Print Model Summary
mdel_summ = model.summary()
print(mdel_summ)

#! Compile The Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

#! Get History
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_dataset
)

#! Evaluate The Model
scores = model.evaluate(tst_dataset)
print(scores)


#! WE CAN CREATE A PROGRESS GRAPH USING HISTORY
print(history)


#! SAVE THE Model

model.export(f"../createdModel/{MODEL_VERSION}")






