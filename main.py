from __future__ import print_function
from __future__ import division

import math

import numpy as np
import pandas as pd
from skimage import color
from skimage.transform import resize
from skimage import io, data
from skimage.io import imread_collection
from matplotlib import pyplot as plt
import os
import random
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import VGG19_Weights, VGG19_BN_Weights
import torchmetrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# reads all images from given directory
# returns list of np.arrays
# each row of an np.array represents an image, transformed from rgb to grayscale
# each column of array represents the grayscale value (between 0-1)
# images are resized to be 100x100 pixels
# if you want to see the image, try -> np.reshape(100, 100)
def images_to_array(dir_name):
    ar = []
    labels = []
    directory = dir_name
    for filename in os.listdir("Vegetable Images/"+directory):
        folder = "Vegetable Images/"+directory+"/"+filename+"/*.jpg"
        col = imread_collection(folder)
        for image in col:
            img = color.rgb2gray(image)
            img_resized = resize(img, (100, 100), anti_aliasing=True)
            ar.append(img_resized.flatten())
            labels.append(filename)

    return np.vstack(ar), np.array(labels)

def one_hot(ar):
    res = []
    veg = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
           'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
    zeros = np.zeros(15, dtype=int)
    for i in ar:
        ind = veg.index(i)
        temp = zeros.copy()
        temp[ind] = 1
        res.append(temp)
    return np.vstack(res)

class Network:
    def __init__(self, structure, epochs=10, alpha=0.1, batch_size=0, regularizer=0.0):
        #Taking the numbers out of the structure list:
        self.sizes = [x for x in structure if isinstance(x, int)]
        #Taking the activation functions out of the structure list:
        self.activations = [x.lower() for x in structure if isinstance(x, str)]
        #Hiperparameters:
        self.epochs = epochs
        self.alpha = alpha
        self.Reg = regularizer
        self.batch_size = batch_size
        #Initializing Weights and bias into Theta:
        self.Thetas = self.initialization()

    def initialization(self): #Randomly start the weights and bias
        # Randomly initializing Thetas, It will be a dictionary containing the Theta of each layer
        Thetas = {}
        for layer in range(len(self.sizes)-1):
            Thetas[f'T{layer+1}'] = np.random.randn(self.sizes[layer]+1, self.sizes[layer+1])/10
        return Thetas #{T1:[...], T2:[...], T3:[...], ...}

    #ACTIVATION FUNCTIONS:

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def dxsigmoid(self, z):
        return np.multiply(self.sigmoid(z),(1-self.sigmoid(z)))

    def softmax(self,z):
        z = np.array(z)
        e_x = np.exp(z - np.max(z, axis=1, keepdims=True))
        return (e_x / e_x.sum(axis=1)[:,None])

    def relu(self, z):
        return np.maximum(0,z)

    def dxrelu(self, z):
        return np.where(z >= 0, 1, 0)

    def tanh(self, z):
        return np.tanh(z)

    def dxtanh(self, z):
        return 1 - np.square(np.tanh(z))

    #TRAINING FUNCTIONS:
    def forward(self, X): #forward propagation
        X = np.matrix(X)
        m = X.shape[0]
        att = self.activations
        Thetas = self.Thetas
        forward_steps = {}
        forward_steps['a1'] = X
        Lastlayer = int(len(self.sizes))
        for layer in range(1,Lastlayer):
            forward_steps[f'z{layer+1}'] = np.dot(forward_steps[f'a{layer}'], Thetas[f'T{layer}'])
            if att[layer-1] == 'sigmoid':
                forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), self.sigmoid(forward_steps[f'z{layer+1}'])), axis=1)
            elif att[layer-1] == 'softmax':
                forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), self.softmax(forward_steps[f'z{layer+1}'])), axis=1)
            elif att[layer-1] == 'relu':
                forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), self.relu(forward_steps[f'z{layer+1}'])), axis=1)
            elif att[layer-1] == 'tanh':
                forward_steps[f'a{layer+1}'] = np.concatenate((np.ones([m,1]), self.tanh(forward_steps[f'z{layer+1}'])), axis=1)
            else:
                print('ERROR')

        h = forward_steps.pop(f'a{Lastlayer}')
        forward_steps['h'] = h[:,1:]

        return forward_steps

    def costFunction(self):
        Y = self.Y
        X = self.X
        Thetas = self.Thetas
        m = self.m
        Reg = self.Reg
        soma_weights = 0
        for i in range(len(Thetas)):
            weights = Thetas[f'T{i+1}']
            weights[0] = 0
            soma_weights += np.sum(weights**2)
        forward_dict = self.forward(X)
        h = forward_dict['h']
        soma = np.sum((np.multiply(-Y , np.log(h)) - np.multiply((1-Y),(np.log(1-h)))))
        J = soma/m + (Reg/(2*m)) * soma_weights
        return J

    def gradients(self, X, Y):
        X = np.matrix(X)
        Y = np.matrix(Y)
        m = X.shape[0]
        Thetas = self.Thetas
        n_layers = len(self.sizes)
        att = self.activations
        Thetas_grad = []

        forward_list = self.forward(X)
        deltas = {}
        deltas[f'delta{n_layers}'] = forward_list['h'] - Y # derivative of the last layer
        for i in range(n_layers-1,1,-1):# 3 ... 2
            if att[i-2] == 'sigmoid':
                deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i+1}'],Thetas[f'T{i}'][1:].T)) , self.dxsigmoid(forward_list[f'z{i}']))
            elif att[i-2] == 'relu':
                deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i+1}'],Thetas[f'T{i}'][1:].T)) , self.dxrelu(forward_list[f'z{i}']))
            elif att[i-2] == 'tanh':
                deltas[f'delta{i}'] = np.multiply((np.dot(deltas[f'delta{i+1}'],Thetas[f'T{i}'][1:].T)) , self.dxtanh(forward_list[f'z{i}']))

        for c in range(len(deltas)):#0 ... 1 ... 2
            BigDelta = np.array(np.dot(deltas[f'delta{c+2}'].T, forward_list[f'a{c+1}']))
            weights = Thetas[f'T{c+1}']
            weights[0] = 0
            grad = np.array(BigDelta + (self.Reg * weights.T))/m
            Thetas_grad.append(grad)
        return Thetas_grad #[T1_grad, T2_grad, T3_grad]

    def accuracy(self, X, Y):
        forward_list = self.forward(X)
        h = forward_list['h']
        y_hat = np.argmax(h, axis=1)[:,None]
        y = np.argmax(Y, axis=1)[:,None]
        return np.mean(y_hat == y)

    def metrics(self, X, Y):
        forward_list = self.forward(X)
        h = forward_list['h']
        y_hat = np.argmax(h, axis=1)[:,None]
        y = np.argmax(Y, axis=1)[:,None]
        res = []
        res.append(accuracy_score(y, y_hat))
        res.append(precision_score(y, y_hat, average="macro", zero_division=0))
        res.append(recall_score(y, y_hat, average="macro"))
        res.append(f1_score(y, y_hat, average="macro"))
        res.append(confusion_matrix(y, y_hat))
        return res
        # print("Accuracy: ", accuracy_score(y, y_hat))
        # print("Precision: ", precision_score(y, y_hat, average="macro", zero_division=0))
        # print("Recall: ", recall_score(y, y_hat, average="macro"))
        # print("F1 score: ", f1_score(y, y_hat, average="macro"))
        # print()

    def train(self, X, Y, x_test, y_test, isTest = False):
        Thetas = self.Thetas
        self.X = X
        self.Y = Y
        self.m = X.shape[0]
        j_history = []
        sec1 = time.time()
        if self.batch_size <= 0:
            b_size = self.m
            print(f'Using batch size: {b_size}..')
        elif isinstance(self.batch_size, int) and (1<= self.batch_size <= self.m):
            b_size = self.batch_size
        else:
            return 'ERROR IN BATCH SIZE'
        for ep in range(self.epochs):
            m = self.m
            a = np.array([0,b_size])
            num = 1 #put a higher number if will use lots of epochs
            self.alpha = (1 / 1 + (0.01 * ep)) * self.alpha # decay
            for i in range(m // b_size):
                inx = a + b_size*i
                grad_list = self.gradients(X[inx[0]:inx[1]], Y[inx[0]:inx[1]])
                for g in range(len(grad_list)):
                    self.Thetas[f'T{g+1}'] = self.Thetas[f'T{g+1}'] - self.alpha*np.array(grad_list[g]).T

            if (ep+1) % num == 0: #
                J = self.costFunction()
                j_history.append(J)
                accu_train = self.accuracy(X,Y)
                accu_test = self.accuracy(x_test,y_test)
                sec2 = time.time()
                tempo_gasto = sec2 - sec1 # time spent
                print(f'Epoch: {ep+1}; Cost: {J:.5f}: Accuracy Train: {accu_train:.5%}; Accuracy Test: {accu_test:.5%}; Time Spent: {tempo_gasto:.2f} s')
                if((ep+1) % 10 == 0 and isTest == False):
                    flag = earlystop(j_history[-10:])
                    if(flag):
                        print("Early stop at ", ep+1)
                        break
        if(isTest):
            return j_history, self.Thetas, self.metrics(x_test, y_test)
        else:
            return j_history, self.Thetas

def earlystop(ar):
    first5 = ar[:5]
    last5 = ar[-5:]
    res = []
    for i in last5:
        flag = all(i > j for j in first5)
        res.append(flag)
    if(False in res):
        return False
    else:
        return True

def transform_data(X, Y):
    m = X.shape[0]       #number of images
    n = X.shape[1] + 1   #number of pixels in each image, +1 because of the bias
    X = np.concatenate((np.ones([m,1]),X), axis=1) # Adding the column of 1's for supporting the bias

    return X, one_hot(Y)

def graph(ar, model):
    plt.plot(ar, 'go-',label='Cost')
    plt.title('Cost X Epochs for function: ' + str(model.activations[0]) + " alpha: " + str(round(model.alpha, ndigits=4)) + " batch-size: " + str(model.batch_size))
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.show()

def visualize(trained_thetas):
    # weights = np.sum(trained_thetas["T1"], axis=1)[1:]
    weights = trained_thetas["T1"][1:, 0]
    weights = weights.reshape(100, 100)
    plt.imshow(weights, cmap="gray")
    plt.title("Vegetable")
    plt.show()

def get_images(trained_thetas):
    layer = list(trained_thetas.keys())[-2]
    veg = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage',
           'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
    for i in range(len(veg)):
        weights = trained_thetas[layer][1:, i]
        pixel = int(math.sqrt(trained_thetas[layer].shape[0] - 1))
        weights = weights.reshape(pixel, pixel)
        plt.imshow(weights, cmap="gray")
        plt.title(veg[i])
        plt.show()

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, isTest = False):
    # Check if we are validating or testing
    if(isTest):
        phaseNext = 'test'
    else:
        phaseNext = 'val'

    since = time.time()

    val_loss_history = []
    best_metrics = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999.0
    prev_loss = 999.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Initialize metrics and confusion matrix
        metric_acc = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=15).to(device)
        metric_pre = torchmetrics.Precision(task='multiclass', average='macro', num_classes=15).to(device)
        metric_rec = torchmetrics.Recall(task='multiclass', average='macro', num_classes=15).to(device)
        metric_f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=15).to(device)
        metric_cm = torchmetrics.ConfusionMatrix(task='multiclass', average='macro', num_classes=15).to(device)

        # Each epoch has a training and validation/test phase
        for phase in ['train', phaseNext]:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # metrics on current batch
                acc = metric_acc(preds, labels.data)
                pre = metric_pre(preds, labels.data)
                rec = metric_rec(preds, labels.data)
                f1 = metric_f1(preds, labels.data)
                cm = metric_cm(preds, labels.data)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # metrics on all batches using custom accumulation
            # and reseting internal state such that metric ready for new data
            acc = metric_acc.compute()
            metric_acc.reset()

            pre = metric_pre.compute()
            metric_pre.reset()

            rec = metric_rec.compute()
            metric_rec.reset()

            f1 = metric_f1.compute()
            metric_f1.reset()

            cm = metric_cm.compute()
            metric_cm.reset()


            # deep copy the model for every 10 epochs and if new model loss is better than previous model
            if phase == phaseNext and (epoch+1)%10 == 0:
                prev_loss = best_loss
                best_loss = epoch_loss
                best_acc = epoch_acc

            # save current model as best model and if testing save metrics of new model
            if phase == phaseNext and best_loss < prev_loss and (epoch+1)%10 == 0:
                best_model_wts = copy.deepcopy(model.state_dict())
                if(isTest):
                    best_metrics.clear()
                    best_metrics.append(acc)
                    best_metrics.append(pre)
                    best_metrics.append(rec)
                    best_metrics.append(f1)
                    best_metrics.append(cm.cpu().numpy())


            if phase == phaseNext:
                val_loss_history.append(epoch_loss)


        print()

        # early stopping when new model loss is worse than previous model
        if (epoch+1)%10 == 0 and best_loss > prev_loss and isTest == False:
            print("Early stop at ", epoch+1)
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print()
    if(isTest):
        print(f"Accuracy on all data: {best_metrics[0]}")
        print(f"Precision on all data: {best_metrics[1]}")
        print(f"Recall on all data: {best_metrics[2]}")
        print(f"F1 score on all data: {best_metrics[3]}")
        print("Confusion matrix on all data:")
        print(best_metrics[4])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract):
    model_ft = models.vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    return model_ft

def get_params_to_update(model, feature_extracting):
    params_to_update = model.parameters()

    if feature_extracting:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    return params_to_update

def graph_cnn(hist, isTest):
    if (isTest):
        case = "Feature Extraction"
    else:
        case = "Finetune"
    plt.plot(hist, 'go-',label='Loss')
    plt.title('Loss X Epochs for '+case)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def visualize_cnn(model, images):
    no_of_layers=0
    conv_layers=[]

    model_children=list(model.cpu().children())

    for child in model_children:
        if type(child)==nn.Conv2d:
            no_of_layers+=1
            conv_layers.append(child)
        elif type(child)==nn.Sequential:
            for layer in child.children():
                if type(layer)==nn.Conv2d:
                    no_of_layers+=1
                    conv_layers.append(layer)

    class_index = 0
    for img in images:
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            results.append(conv_layers[i](results[-1]))
        outputs = results
        num_layer = 15
        plt.figure(figsize=(10, 2))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(class_names[class_index])
        for i, filter in enumerate(layer_viz):
            if i == 16:
                break
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.show()
        plt.close()
        class_index += 1

if __name__ == '__main__':
    # PART 1
    # read images and convert them from rgb to grayscale
    print("Reading data starts...")
    train_x, train_y = images_to_array("train")
    test_x, test_y = images_to_array("test")
    validation_x, validation_y = images_to_array("validation")
    print("Reading data ended.")

    train_x, train_y = transform_data(train_x, train_y)
    validation_x, validation_y = transform_data(validation_x, validation_y)
    test_x, test_y = transform_data(test_x, test_y)

    # parameters = structure (no-hidden layer, 1-hidden, 2-hiddden),
    # activation function (sigmoid, relu, tanh),
    # alpha (low = 0.005, mid = 0.0125, high = 0.02),
    # batch-size (low = 16, mid = 72, high = 128)
    cases = {
        "activation": ["sigmoid", "relu", "tanh"],
        "alpha": [0.005, 0.0125, 0.02],
        "batch": [16, 72, 128]}

    nn_models = {"no-layer": "", "1-layer": "", "2-layer": ""}
    min_cost = [999, 999, 999]
    thetas = np.array([{},{},{}])

    for func in cases["activation"]:
        for alpha in cases["alpha"]:
            for batch_size in cases["batch"]:
                model1 = Network([10000, func, 15, "softmax", 15], epochs=100, alpha=alpha, batch_size=batch_size)
                j_history1, trained_thetas1 = model1.train(train_x, train_y, validation_x, validation_y)
                if(j_history1[-1] < min_cost[0]):
                    min_cost[0] = j_history1[-1]
                    nn_models["no-layer"] = model1
                    thetas[0] = trained_thetas1
                graph(j_history1, model1)

                model2 = Network([10000, func,225, func, 15, "softmax", 15], epochs=100, alpha=alpha, batch_size=batch_size)
                j_history2, trained_thetas2 = model2.train(train_x, train_y, validation_x, validation_y)
                if(j_history2[-1] < min_cost[1]):
                    min_cost[1] = j_history2[-1]
                    nn_models["1-layer"] = model2
                    thetas[1] = trained_thetas2
                graph(j_history2, model2)

                model3 = Network([10000, func,500, func, 225, func, 15, "softmax", 15], epochs=100, alpha=alpha, batch_size=batch_size)
                j_history3, trained_thetas3 = model3.train(train_x, train_y, validation_x, validation_y)
                if(j_history3[-1] < min_cost[2]):
                    min_cost[2] = j_history3[-1]
                    nn_models["2-layer"] = model3
                    thetas[2] = trained_thetas3
                graph(j_history3, model3)

    # model = Network([10000, "sigmoid", 15, "softmax", 15], epochs=3, alpha=0.02, batch_size=16)
    # j_history, trained_thetas = model.train(train_x, train_y, validation_x, validation_y)
    #
    # model2 = Network([10000, "sigmoid",225, "sigmoid", 15, "softmax", 15], epochs=3, alpha=0.01, batch_size=72)
    # j_history2, trained_thetas2 = model2.train(train_x, train_y, validation_x, validation_y)
    #
    # model3 = Network([10000, "sigmoid",225, "sigmoid", 225, "sigmoid", 15, "softmax", 15], epochs=3, alpha=0.005, batch_size=128)
    # j_history3, trained_thetas3 = model3.train(train_x, train_y, validation_x, validation_y)
    #
    # thetas[0] = trained_thetas
    # thetas[1] = trained_thetas2
    # thetas[2] = trained_thetas3

    # nn_models["no-layer"] = model
    # nn_models["1-layer"] = model2
    # nn_models["2-layer"] = model3

    np.save("thetas.npy", thetas)
    metric_results=[]
    for model in nn_models.values():
        j_history, trained_thetas, mets = model.train(train_x, train_y, test_x, test_y, isTest=True)
        graph(j_history, model)
        print(mets[4])
        mets = mets[:4]
        metric_results.append(mets)
    mux = pd.MultiIndex.from_product([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
    dict_res = {"no-layer":metric_results[0], "1-layer":metric_results[1], "2-layer":metric_results[2]}
    df_res = pd.DataFrame.from_dict(dict_res, orient='index', columns = mux)
    print(df_res)


    thetas = np.load("thetas.npy", allow_pickle=True)


    print("Visualization of classes for model 1")
    get_images(thetas[0])

    print("Visualization of classes for model 2")
    get_images(thetas[1])

    print("Visualization of classes for model 3")
    get_images(thetas[2])


    # PART 2

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./Vegetable Images"

    # Number of classes in the dataset
    num_classes = 15

    # Batch size for training (change depending on how much memory you have)
    batch_size = 32

    # Number of epochs to train for
    num_epochs = 30

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Custom input size for our model
    input_size = 100

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {"train": datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"]),
                        "val": datasets.ImageFolder(os.path.join(data_dir, "validation"), data_transforms["val"]),
                        "test": datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["val"])}

    # Get class names
    class_names = image_datasets['train'].classes

    # Limit the data size to %10 of original size
    reduced_train = []
    reduced_val = []
    for i in range(15):
        reduced_train.extend(range(i*1000,i*1000+ 100))
        reduced_val.extend(range(i*200,i*200+ 20))

    image_datasets["train"] = torch.utils.data.Subset(image_datasets["train"], reduced_train)
    image_datasets["val"] = torch.utils.data.Subset(image_datasets["val"], reduced_val)
    image_datasets["test"] = torch.utils.data.Subset(image_datasets["test"], reduced_val)

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get specific images from test file and transform them to use in visualization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(100),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get only first images of every class
    images = []
    for filename in os.listdir("Vegetable Images/test/"):
        for imgname in os.listdir("Vegetable Images/test/"+filename):
            img = io.imread("./Vegetable Images/test/"+filename+"/"+imgname)
            img = np.array(img)
            img = transform(img)
            img = img.unsqueeze(0)
            images.append(img)
            break

    # Initialize the model for this run
    model_ft = initialize_model(num_classes, feature_extract)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Get parameters to update in model
    params_to_update = get_params_to_update(model_ft, feature_extract)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.005, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate for finetune
    model_ft, hist_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    torch.cuda.empty_cache()

    # Test for finetune
    model_ft_test, hist_ft_test = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, isTest = True)

    visualize_cnn(model_ft_test, images)

    graph_cnn(hist_ft_test, feature_extract)

    torch.cuda.empty_cache()

    feature_extract = True

    # Initialize the model for this run
    model_fe = initialize_model(num_classes, feature_extract)

    # Send the model to GPU
    model_fe = model_fe.to(device)

    # Get parameters to update in model
    params_to_update = get_params_to_update(model_fe, feature_extract)

    # Observe that all parameters are being optimized
    optimizer_fe = optim.SGD(params_to_update, lr=0.005, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate for feature extract
    model_fe, hist_fe = train_model(model_fe, dataloaders_dict, criterion, optimizer_fe, num_epochs=num_epochs)

    torch.cuda.empty_cache()

    # Test for feature extract
    model_fe_test, hist_fe_test = train_model(model_fe, dataloaders_dict, criterion, optimizer_fe, num_epochs=num_epochs, isTest = True)

    visualize_cnn(model_fe_test, images)

    graph_cnn(hist_fe_test, feature_extract)
