from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
#loading CNN3D classes
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, InputLayer, BatchNormalization, Dropout, GlobalAveragePooling3D, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras import layers
import numpy as np
import keras
import os
from keras.layers import Convolution2D
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Bidirectional, GRU, Conv1D, MaxPooling1D, RepeatVector#loading GRU, bidriectional, and CNN
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Human Behaviour") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
precision = []
recall = []
fscore = []
accuracy = []

#defining class labels
labels = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']

def uploadDataset():
    global filename, dataset, labels, values,X,Y
    text.delete('1.0', END)
    filename1 = filedialog.askopenfilename(initialdir = "Dataset")
    text.insert(END,'Dataset loaded\n\n')
    X = pd.read_csv("Dataset/X_train.txt", header=None, delim_whitespace=True)
    text.insert(END,str(X)+"\n")
    filename2 = filedialog.askopenfilename(initialdir = "Dataset")
    Y = pd.read_csv("Dataset/y_train.txt", header=None, delim_whitespace=True)
    text.insert(END,str(Y)+"\n")
    names, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.show()

def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, scaler
    text.delete('1.0', END)
    X = X.values
    Y = Y.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 17, 11, 3, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing"+"\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0])+"\n")
    
   

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(4, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def trainCNN2D():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    text.delete('1.0', END)
    #train existing CNN algorithm which will use many parameters for training and can increase computation complexity
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], (X_train.shape[3] * X_train.shape[4])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], (X_test.shape[3] * X_test.shape[4]))) 
    cnn_model = Sequential()
    #define cnn2d layer with 3 number of inout neurons and to filter dataset features
    cnn_model.add(Convolution2D(3, (1 , 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    #collect filtered features from CNN2D layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #defining another layer t further optimize features
    cnn_model.add(Convolution2D(3, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Flatten())
    #define output layer
    cnn_model.add(Dense(units = 16, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(cnn_model.summary()) 
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train1, y_train1, batch_size = 200, epochs = 20, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #perform prediction on human behaviour on test data using CNN existing model
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Existing CNN Model", predict, y_test1)

def trainCNN3D():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    text.delete('1.0', END)
    inputs = keras.Input((X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
    x = layers.Conv3D(filters=7, kernel_size=3, activation="relu")(inputs)#creating CNN3D layer with 7 neurons for data filter
    x = layers.MaxPool3D(pool_size=1)(x) #pool layer to collect filterd features from CNN3D layer
    x = layers.BatchNormalization()(x) #normalizing features
    x = layers.Conv3D(filters=7, kernel_size=1, activation="relu")(x)#another layer to optimze module using space time
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=7, kernel_size=1, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=1, activation="relu")(x)#cnn layer for separable convolution module
    x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)#defining global average pooling
    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units=y_train.shape[1], activation="softmax")(x)
    mdn_model = keras.Model(inputs, outputs, name="3dcnn") #create model
    mdn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #displaying propose model complexity
    print(mdn_model.summary())
    if os.path.exists("model/mdn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/mdn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = mdn_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/mdn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        mdn_model.load_weights("model/mdn_weights.hdf5")
        
    predict = mdn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)#calculate metrics
    calculateMetrics("Propose MDN Model", predict, y_test1)

def trainBiGRU():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    text.delete('1.0', END)
    extension_model = Sequential()
    extension_model.add(Convolution2D(2, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Convolution2D(1, (1, 1), activation = 'relu'))
    extension_model.add(MaxPooling2D(pool_size = (1, 1)))
    extension_model.add(Flatten())
    extension_model.add(RepeatVector(2))
    #adding bidirectional + GRU to CNN layer
    extension_model.add(Bidirectional(GRU(1, activation = 'relu')))
    extension_model.add(Dense(units = 1, activation = 'relu'))
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train the model
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(extension_model.summary()) 
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model = load_model("model/extension_weights.hdf5")

    predict = extension_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("Extension Hybrid Model CNN + GRU + Bidirectional", predict, y_test1)

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value


def graph():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    existing_acc, existing_loss = values("model/cnn_history.pckl", "accuracy", "loss")
    propose_acc, propose_loss = values("model/mdn_history.pckl", "accuracy", "loss")
    extension_acc, extension_loss = values("model/extension_history.pckl", "accuracy", "loss")
        
    plt.figure(figsize=(6,4))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(existing_acc, 'ro-', color = 'gray')
    plt.plot(propose_acc, 'ro-', color = '#fb6f92')
    plt.plot(extension_acc, 'ro-', color = 'blue')
    plt.legend(['Existing CNN2D Model', 'Propose MDN CNN3D Model', 'Extension Ensemble CNN + Bidirectional + GRU Model'], loc='lower left')
    plt.title('All Algorithm Training Accuracy Graph')
    plt.show()

def cGraph():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    df = pd.DataFrame([
        ['Existing CNN2D', 'Accuracy', accuracy[0]], 
        ['Existing CNN2D', 'Precision', precision[0]], 
        ['Existing CNN2D', 'Recall', recall[0]], 
        ['Existing CNN2D', 'FSCORE', fscore[0]],

        ['Propose MDN CNN3D', 'Accuracy', accuracy[1]], 
        ['Propose MDN CNN3D', 'Precision', precision[1]], 
        ['Propose MDN CNN3D', 'Recall', recall[1]], 
        ['Propose MDN CNN3D', 'FSCORE', fscore[1]],

        ['Extension CNN + Bidirectional + GRU', 'Accuracy', accuracy[2]], 
        ['Extension CNN + Bidirectional + GRU', 'Precision', precision[2]], 
        ['Extension CNN + Bidirectional + GRU', 'Recall', recall[2]], 
        ['Extension CNN + Bidirectional + GRU', 'FSCORE', fscore[2]],
    ], columns=['Algorithms', 'Parameters', 'Value'])

    # Setting Style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Creating Barplot
    ax = sns.barplot(x="Parameters", y="Value", hue="Algorithms", data=df, palette="viridis", edgecolor="black")

    # Customizing Appearance
    plt.title("Performance Comparison of Algorithms", fontsize=14, fontweight='bold')
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Scores", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title="Algorithms", fontsize=10, title_fontsize=12)

    plt.show()

def predict():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values,extension_model,testY,X_test1
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    testData = pd.read_csv(filename, header=None, delim_whitespace=True)
    testData = testData.values
    indices = np.arange(testData.shape[0])
    np.random.shuffle(indices)#shuffling test data to select random 10 records
    testData = testData[indices]
    testData = testData[0:10,0:testData.shape[1]]#select 10 records
    testData1 = np.reshape(testData, (testData.shape[0], 17, 11, 3)) #convert test data as per CNN model
    predict = extension_model.predict(testData1)#perform prediction on test dtaa
    for i in range(len(predict)):
        pred = np.argmax(predict[i])
        text.insert(END,"Test Data : "+str(testData[i][0:30])+" Predicted Activity ===> "+labels[pred-1]+"\n")
    


font = ('times', 16, 'bold')
title = Label(main, text='Human Behaviour Recognition Based on Multiscale Convolutional Neural Network')
title.config(bg='#003049', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(relx=0.5, y=30, anchor='center')

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=250)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
title.place(relx=0.5, y=30, anchor='center')
text.place(x=60, y=200)

text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=180,y=100)
processButton.config(font=font1)

cnnButton = Button(main, text="Run Existing CNN2D", command=trainCNN2D)
cnnButton.place(x=440,y=100)
cnnButton.config(font=font1)

cnn3DButton = Button(main, text="Run Puposed CNN3D", command=trainCNN3D)
cnn3DButton.place(x=660,y=100)
cnn3DButton.config(font=font1)

biGRUButton = Button(main, text="Run Extension CNN + GRU + Bidirectional", command=trainBiGRU)
biGRUButton.place(x=860,y=100)
biGRUButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

cgraphButton = Button(main, text="Comparision Graph", command=cGraph)
cgraphButton.place(x=220,y=150)
cgraphButton.config(font=font1)

predictButton = Button(main, text="Predict Human Behaviour", command=predict)
predictButton.place(x=440,y=150)
predictButton.config(font=font1)

main.config(bg='#669bbc')
main.mainloop()

