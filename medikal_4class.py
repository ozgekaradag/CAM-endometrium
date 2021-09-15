import glob
import pdb
import tensorflow as tf
import keras
from tensorflow import keras
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import cv2
import pickle
import MACOSFile

img_width = 224
img_height = 224

class_names = ['EA', 'EH', 'EP', 'NE']
#class_names = ['benign', 'malignant']
num_classes = len(class_names)

EA = glob.glob('./dataset/EA/*.JPG')
EH = glob.glob('./dataset/EH/*.JPG')
EP = glob.glob('./dataset/EP/*.JPG')
NE = glob.glob('./dataset/NE/*.JPG')
##benign = glob.glob('./dataset2/benign/*.JPG')
##malignant= glob.glob('./dataset2/malignant/*.JPG')

data = []
labels = []

for i in EA:
    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
    image= np.array(image)
    data.append(image)
    labels.append(0)

for i in EH:
    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
    image=np.array(image)
    data.append(image)
    labels.append(1)

for i in EP:
    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
    image=np.array(image)
    data.append(image)
    labels.append(2)

for i in NE:
    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
    image=np.array(image)
    data.append(image)
    labels.append(3)

##for i in benign:
##    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
##    image= np.array(image)
##    data.append(image)
##    labels.append(0)
##
##for i in malignant:
##    image= load_img(i, color_mode='rgb', target_size= (img_width, img_height))
##    image=np.array(image)
##    data.append(image)
##    labels.append(1)
    
data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f'X_train.shape >> {X_train.shape}\nX_test.shape >> {X_test.shape}')

y_train = to_categorical(y_train, num_classes= num_classes)
y_test = to_categorical(y_test, num_classes= num_classes)

##### save data to be sure that same set of train/test pair is used at each run ######
import os.path

file_path = "train_data.pkl"
n_bytes = 2**31
max_bytes = 2**31 - 1
data = bytearray(n_bytes)

## write train
bytes_out = pickle.dumps([X_train,y_train])
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

file_path = "test_data.pkl"
## write test
bytes_out = pickle.dumps([X_test,y_test])
with open(file_path,'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])

## load data
#X_train, y_train=MACOSFile.pickle_load("train_data.pkl")
#X_test, y_test = MACOSFile.pickle_load("test_data.pkl")

##########################################################################################
#####veri arttırma flip ediyor. bulduğumuz çalışmada yapılıyordu.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # image preprocessing function
    horizontal_flip=False, #True, 
    vertical_flip=False #True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input # image preprocessing function
    )



def build_model():
    input_tensor = Input(shape=(img_height, img_width, 3))
    resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    
   # for layer in (resnet50.layers):
   #     layer.trainable = True #!!
   #     if layer.name.startswith('bn'):
   #         layer.call(layer.input, training=False)

    x = GlobalAveragePooling2D(name='avg_pool')(resnet50.output)
    x = Dense(512, activation='relu')(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(resnet50.input, predictions)
    
    return model


model = build_model()
# model.summary()
#plot_model(model,to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#pdb.set_trace()

model.compile(
  optimizer= Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
  loss='categorical_crossentropy',
  metrics=[categorical_accuracy]
  )


early_stopper = EarlyStopping(
    monitor='val_categorical_accuracy', 
    min_delta=1e-4, 
    patience=50,
    restore_best_weights=True
    )

reduce_lr = ReduceLROnPlateau(
    monitor='categorical_accuracy', 
    factor=0.5,
    cooldown=0, 
    patience=5, 
    min_lr=0.5e-6
    )

model_checkpoint = ModelCheckpoint(
    filepath = './weights/{epoch:02d}-{categorical_accuracy:.4f}-{val_loss:.4f}-{val_categorical_accuracy:.4f}_4class.h5', 
    monitor='val_categorical_accuracy', 
    save_best_only=True,
    verbose=2,
    save_weights_only=True, 
    mode='max'
    )

callbacks = [early_stopper, reduce_lr, model_checkpoint]


#train_datagen.fit(X_train)


##history = model.fit(
##    train_datagen.flow(X_train, y_train),
##    steps_per_epoch=len(X_train) / 32,
##    epochs=100,
##    validation_data=val_datagen.flow(X_test, y_test),
##    callbacks=[callbacks]
##    )
##model.save('./weights/final_4class.h5')

model = load_model('./weights/final_4class.h5')
#model.load_weights('./weights/48-0.9743-2.1275-0.6203.h5')

##history = model.fit(
##    train_datagen.flow(X_train, y_train),
##    steps_per_epoch=len(X_train) / 32,
##    epochs=100,
##    validation_data=val_datagen.flow(X_test, y_test),
##    callbacks=[callbacks]
##    )
##model.save('./weights/final_4class.h5')
'''
plt.figure(figsize=(15, 4))
plt.subplot(121),
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
'''

y_pred = model.predict(preprocess_input(X_test))
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)



report = classification_report(y_true, y_pred, target_names= class_names, output_dict=True)
print(pd.DataFrame(report).transpose())

cm = confusion_matrix(y_true, y_pred)
print("\n",cm)





def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_array, cam, emphasize=False, extract=False):

    heatmap = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)

    if extract:
        _, mask = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
        result = cv2.bitwise_or(img_array, img_array, mask= mask)
        return result
        
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_array
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img_rgb

## score-cam

def ScoreCam(model, img_array, layer_name = 'conv5_block3_out', max_N=-1):

    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam


## grad-cam

def GradCam(model, img_array, layer_name = 'conv5_block3_out', pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


## grad-cam++

def GradCamPlus(model, img_array, layer_name='conv5_block3_out', label_name=None, category_id=None):

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_array)
                if category_id==None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)
    
    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_CAM_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap


# test kümemiz 661 elemandan oluşuyor
score_results = np.zeros((661,5))
grad_results = np.zeros((661,5))
gradplus_results = np.zeros((661,5))


score_average_drop = 0
grad_average_drop = 0
gradplus_average_drop = 0

score_average_increase = 0
grad_average_increase = 0
gradplus_average_increase = 0

for i in [81,82]: #range(661):
    img = X_test[i]
    img_array = np.expand_dims(img, axis=0)
    #plt.imshow(img_array.reshape(224,224,3))
    #plt.show()
    cv2.imwrite('sample_image_'+str(i)+'_wlab_'+str(y_test[i])+'.bmp',img_array.reshape(224,224,3))
    #plt.savefig('\sample_images_cams\image'+str(i)+'_wlab_'+str(y_test[i]))
    img_array = preprocess_input(img_array)
    orig_img_pred = model.predict(img_array)
    print(f"X_test[{i}] = {class_names[np.argmax(y_test[i])]}\nFull image > {class_names[np.argmax(orig_img_pred[0])]} >> {np.max(orig_img_pred[0])}")


    
    #scorecam
    score_cam = ScoreCam(model,img_array)
    score_cam_extract = superimpose(img, score_cam, extract=True)
    score_cam_extract_array = np.expand_dims(score_cam_extract, axis=0)
    #plt.imshow(score_cam_extract_array.reshape(224,224,3))
    #plt.show()
    cv2.imwrite('sample_image_'+str(i)+'_scorecam.bmp',score_cam_extract_array.reshape(224,224,3))
    score_cam_extract_array = preprocess_input(score_cam_extract_array)
    #plt.imshow(score_cam_extract_array.reshape(224,224,3))
    #plt.show()
    score_cam_pred = model.predict(score_cam_extract_array)
    print(f"Score-Cam > {class_names[np.argmax(score_cam_pred[0])]} >> {np.max(score_cam_pred[0])}")

    score_results[i][0] = np.argmax(y_test[i]) # true class
    score_results[i][1] = np.max(orig_img_pred[0]) # predict full image result
    score_results[i][2] = np.argmax(orig_img_pred[0]) # predict full image class
    score_results[i][3] = np.max(score_cam_pred[0]) # predict score-cam result
    score_results[i][4] = np.argmax(score_cam_pred[0]) # predict score-cam class

    score_average_drop +=np.max([0, score_results[i][1]-score_cam_pred[0][int(score_results[i][2])]]) / score_results[i][1]
    print(f"Score-Cam Average Drop > {score_average_drop}")
    if score_cam_pred[0][int(score_results[i][2])]>score_results[i][1]:
        score_average_increase +=1


    #gradcam
    grad_cam= GradCam(model, img_array)
    grad_cam_extract = superimpose(img, grad_cam, extract=True)
    grad_cam_extract_array = np.expand_dims(grad_cam_extract, axis=0)
    #plt.imshow(grad_cam_extract_array.reshape(224,224,3))
    #plt.show()
    cv2.imwrite('sample_image_'+str(i)+'_gradcam.bmp',grad_cam_extract_array.reshape(224,224,3))
    grad_cam_extract_array = preprocess_input(grad_cam_extract_array)

    grad_cam_pred = model.predict(grad_cam_extract_array)
    print(f"Grad-Cam > {class_names[np.argmax(grad_cam_pred[0])]} >> {np.max(grad_cam_pred[0])}")

    grad_results[i][0] = np.argmax(y_test[i]) # true class
    grad_results[i][1] = np.max(orig_img_pred[0]) # predict full image result
    grad_results[i][2] = np.argmax(orig_img_pred[0]) # predict full image class
    grad_results[i][3] = np.max(grad_cam_pred[0]) # predict grad-cam result
    grad_results[i][4] = np.argmax(grad_cam_pred[0]) # predict gradcam class

    grad_average_drop += np.max([0, grad_results[i][1]-grad_cam_pred[0][int(grad_results[i][2])]]) / grad_results[i][1]
    print(f"GradCam Average Drop > {grad_average_drop}")
    if grad_cam_pred[0][int(grad_results[i][2])]>grad_results[i][1]:
        grad_average_increase+=1


    #gradcam++
    grad_cam_plus = GradCamPlus(model, img_array)
    grad_cam_plus_extract = superimpose(img, grad_cam_plus, extract=True)
    grad_cam_plus_extract_array = np.expand_dims(grad_cam_plus_extract, axis=0)
    #plt.imshow(grad_cam_plus_extract_array.reshape(224,224,3))
    #plt.show()
    cv2.imwrite('sample_image_'+str(i)+'_gradcam_plus.bmp',grad_cam_plus_extract_array.reshape(224,224,3))
    grad_cam_plus_extract_array = preprocess_input(grad_cam_plus_extract_array)

    grad_cam_plus_pred = model.predict(grad_cam_plus_extract_array)
    print(f"Grad-Cam++ > {class_names[np.argmax(grad_cam_plus_pred[0])]} >> {np.max(grad_cam_plus_pred[0])}\n")

    gradplus_results[i][0] = np.argmax(y_test[i]) # true class
    gradplus_results[i][1] = np.max(orig_img_pred[0]) # predict full image result
    gradplus_results[i][2] = np.argmax(orig_img_pred[0]) # predict full image class
    gradplus_results[i][3] = np.max(grad_cam_plus_pred[0]) # predict gradcamplus result
    gradplus_results[i][4] = np.argmax(grad_cam_plus_pred[0]) # predict gradcamplus class

    gradplus_average_drop += np.max([0, gradplus_results[i][1]-grad_cam_plus_pred[0][int(gradplus_results[i][2])]]) / gradplus_results[i][1]
    print(f"GradCam++ Average Drop > {gradplus_average_drop}")
    if grad_cam_plus_pred[0][int(gradplus_results[i][2])]>gradplus_results[i][1]:
        gradplus_average_increase +=1



###sonuçları csv ile daha rahat paylaşırız daha açıklayıcı olur
##
##df = pd.DataFrame(grad_results)
##df.columns=['true class', 'full-image confidence', 'w fi predict class', 'Grad confidence', 'w sc predict class']
##df['true class'].replace({0.0:'EA', 1.0:'EH', 2.0:'EP', 3.0:'NE'},inplace=True)
##df['w fi predict class'].replace({0.0:'EA', 1.0:'EH', 2.0:'EP', 3.0:'NE'},inplace=True)
##df['w sc predict class'].replace({0.0:'EA', 1.0:'EH', 2.0:'EP', 3.0:'NE'},inplace=True)
##
##df.to_csv('sonuc.csv', index=False)

print(np.shape(orig_img_pred))


##score_average_drop = 0
##for i in range(1):
##    score_average_drop +=np.max([0, score_results[i][1]-score_cam_pred[0][int(score_results[i][2])]]) / score_results[i][1]
##    #np.max(orig_img_pred[i][0])==np.argmax(y_test[i]) and score_results[i][3]==np.argmax(y_test[i]):
##    #    score_average_drop += np.maximum(0, (score_results[i][1] - score_results[i][3])) / score_results[i][1]
    
score_average_drop *=100
print("Score-Cam için Average Drop >>", score_average_drop / 661)
print("Score-Cam için Average Increase >>", score_average_increase/661)



##grad_average_drop = 0
##for i in range(1):
##    grad_average_drop += np.max([0, grad_results[i][1]-grad_cam_pred[0][int(grad_results[i][2])]]) / grad_results[i][1]
##    
grad_average_drop *= 100
print("Grad-Cam için Average Drop >>", grad_average_drop / 661)
print("Grad-Cam için Average Increase >>", grad_average_increase/661)

##gradplus_average_drop = 0
##for i in range(1):
##    gradplus_average_drop += np.max([0, gradplus_results[i][1]-grad_cam_plus_pred[0][int(gradplus_results[i][2])]]) / gradplus_results[i][1]
##
##    
gradplus_average_drop *=100
print("Grad-Cam++ için Average Drop >>", gradplus_average_drop / 661)
print("Grad-Cam++ için Average Increase >>", gradplus_average_increase/661)
