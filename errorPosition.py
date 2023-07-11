#importing the libraries
import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
#defining the network
from torch import nn
from torch.nn import functional as F
# defining dataset class
from torch.utils.data import Dataset, DataLoader
from statistics import mean


def generateTupleOverlap(sampleX, sampleY):
    tempOverlap = []
    for i in range(len(sampleX)):
        tempOverlap.append((sampleX[i], sampleY[i]))
    return tempOverlap


def calculatePearsonCoefficient(overlap):
    '''
                            A - B
        pearson = -------------------------
                  sqrt(C - D) * sqrt(E - F)
    '''

    N = len(overlap)
    Xvalues = []
    Yvalues = []
    for i in range(len(overlap)):
        Xvalues.append(overlap[i][0])
        Yvalues.append(overlap[i][1])
    Xmean = sum(Xvalues) / len(Xvalues)
    Ymean = sum(Yvalues) / len(Yvalues)

    #PART A
    AValues = []
    for i in range(len(overlap)):
        temp = overlap[i][0] * overlap[i][1]
        AValues.append(temp)
    A = sum(AValues)

    #PART B
    B = N * Xmean * Ymean

    #PART C
    CValues = []
    for i in range(len(Xvalues)):
        temp = Xvalues[i] * Xvalues[i]
        CValues.append(temp)
    C = sum(CValues)

    #PART D
    D = N * (Xmean * Xmean)

    #PART E
    EValues = []
    for i in range(len(Yvalues)):
        temp = Yvalues[i] * Yvalues[i]
        EValues.append(temp)
    E = sum(EValues)

    #PART F
    F = N * (Ymean * Ymean)

    #Final Calc
    top = A - B
    BottomPartOne = math.sqrt(C - D)
    BottomPartTwo = math.sqrt(E - F)

    PearsonCoefficient = top / (BottomPartOne * BottomPartTwo)

    return PearsonCoefficient


def calculateTValue(overlap, pearson_coefficient):
    N = len(overlap)
    R = pearson_coefficient
    top = R * (math.sqrt(N-2))
    bottom = math.sqrt(1 - (R * R))
    tValue = top / bottom
    return tValue


def calculateTValues(sampleX, sampleY):
    prepX = []
    prepY = []

    for i in range((len(sampleX) + len(sampleY))):
        x = 0
        y = 0
        if i >= len(sampleY):
            x = sampleX[i - len(sampleY)]
        if i <= len(sampleY) - 1:
            y = sampleY[i]
        prepX.append(x)
        prepY.append(y)

    allOverlaps = []
    npX = np.array(prepX)
    npY = np.array(prepY)

    allOverlaps.append(generateTupleOverlap(npX, npY))

    tempSampleY = npY
    while tempSampleY[-1] == 0:
        tempSampleY = np.roll(tempSampleY, 1)
        allOverlaps.append(generateTupleOverlap(npX, tempSampleY))

    tempSampleX = npX
    while tempSampleX[0] == 0:
        tempSampleX = np.roll(tempSampleX, -1)
        allOverlaps.append(generateTupleOverlap(tempSampleX, tempSampleY))

    #allOverlaps is an array of arrays of tuples containing every possible overlap of the two given samples
    #where the first array is the starting position (no overlap) and the last array is the ending position (no overlap)

    allOverlaps.pop(0)
    allOverlaps.pop(-1)

    newAllOverlaps = []

    for i in range(len(allOverlaps)):
        tempArray = []
        stringBuffer = ""
        for j in range(len(allOverlaps[i])):
            if allOverlaps[i][j][0] == 0:
                stringBuffer = "ignore"
            elif allOverlaps[i][j][1] == 0:
                stringBuffer = "ignore"
            else:
                tempArray.append(allOverlaps[i][j])
        newAllOverlaps.append(tempArray)


    #newAllOverlaps now is an array of arrays of tuples containing every possible useful overlap of two samples

    #NEXT: calculate pearson coefficients and t values
    #AFTER: return the array of t values


    '''this is the code for calculation the pearson and tvalue for a single overlap section'''
    '''temp_p = calculatePearsonCoefficient(newAllOverlaps[2])
    temp_a = calculateTValue(newAllOverlaps[2], temp_p)'''


    '''for over in newAllOverlaps:
        print(over)'''



    '''THIS SECTION IS THERE TO REMOVE SCENARIOS THAT CANNOT BE CALCULATED'''
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    '''END SECTION'''

    '''THIS SECTION IS THERE TO REMOVE STATISTICAL ANOMALIES'''
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    newAllOverlaps.pop(-1)
    newAllOverlaps.pop(0)
    '''END SECTION'''

    pearsons = []
    tValues = []

    for over in newAllOverlaps:
        temp_p = calculatePearsonCoefficient(over)
        temp_t = calculateTValue(over, temp_p)
        pearsons.append(temp_p)
        tValues.append(temp_t)

    # pearsons is a list of all of the pearson coefficients for all of the sample overlap posistions
    # tValues is their corresponding t values

    # sample y moves over x starting from the left most position and wokring to the right most

    # lag indicated by index representing position from starting position, i.e 0th index is sample y placed before sample
    # x with no overlap. Thus 1st index (index=1) is sample y placed before sample x with 1 overlap

    lagAmount = []

    for i in range(len(pearsons)):
        lagAmount.append(i+3)

    return tValues, lagAmount, pearsons


class MLP_Network(nn.Module):
    def __init__(self, input_shape, hidden_1_shape, hidden_2_shape, output_shape):
        super(MLP_Network, self).__init__()
        self.hid1 = nn.Linear(input_shape, hidden_1_shape)  # 4-(8-8)-1
        #self.hid2 = nn.Linear(hidden_1_shape, hidden_2_shape)  # 4-(8-8)-1
        self.oupt = nn.Linear(hidden_1_shape, output_shape)

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        #nn.init.xavier_uniform_(self.hid2.weight)
        #nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        #z = torch.tanh(self.hid2(z))
        z = torch.sigmoid(self.oupt(z))
        return z


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length



input_file = open("neural_data/input_data.txt", "r")
input_lines = input_file.readlines()
input_file.close()

target_file = open("neural_data/output_data_binary.txt", "r")
target_lines = target_file.readlines()
target_file.close()

target_file_2 = open("neural_data/output_data_position.txt", "r")
target_lines_2 = target_file_2.readlines()
target_file_2.close()

def cleanData(lines):
    split_lines = []

    for lin in lines:
        split_lines.append(lin.split(','))

    #print(split_lines)

    for i in range(len(split_lines)):
        for j in range(len(split_lines[i])):
            split_lines[i][j] = split_lines[i][j].replace("\n", "")

    for i in range(len(split_lines)):
        for j in range(len(split_lines[i])):
            split_lines[i][j] = float(split_lines[i][j])

    return split_lines


jim = cleanData(input_lines)
bob = cleanData(target_lines)
barbara = cleanData(target_lines_2)

#jim is input bob is output
import random


base = []
for i in range(0,100):
    base.append(0)

advanced_labels = []
for idx, ele in enumerate(barbara):
    temp = copy.deepcopy(base)
    for val in ele:
        temp[int(val)] = 1
    advanced_labels.append(temp)

print(advanced_labels[31])


'''features = []
for value in jim:
    temp = value[:102]
    features.append(temp)'''



random.Random(4).shuffle(jim)
random.Random(4).shuffle(advanced_labels)

tenk_fea = jim[0:3000]
tenk_lab = advanced_labels[0:3000]

twopoin_fea = jim[3000:3867]
twopoin_lab = advanced_labels[3000:3867]

err_counter = 0
non_counter = 0
#PERCENTAGE OF EACH
for lab in twopoin_lab:
    if lab[0] == 1.0:
        err_counter = err_counter + 1
    else:
        non_counter = non_counter + 1
#------------------

print(err_counter)
print(non_counter)


trainset = dataset(tenk_fea, tenk_lab)
# DataLoader
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)


#network parameters
input_dim = 396 #102
hidden_1_dim = 500
hidden_2_dim = 200
output_dim = 100
#hyper parameters
learning_rate = 0.001
epochs = 1000
# Model , Optimizer, Loss
model = MLP_Network(input_shape=input_dim, hidden_1_shape=hidden_1_dim, hidden_2_shape=hidden_2_dim, output_shape=output_dim)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

loss_fn = nn.BCELoss()
print(model)

losses = []
accur = []
vali_losses = []
total_err_accuracy = []
total_general_accuracy = []
for i in range(epochs):
    for j, (x_train, y_train) in enumerate(trainloader):
        # calculate output
        output = model(x_train)

        # calculate loss
        loss = loss_fn(output, y_train.reshape(-1, 100))

        # accuracy
        predicted = model(torch.tensor(twopoin_fea, dtype=torch.float32))
        #acc = (predicted.reshape(-1).detach().numpy().round() == twopoin_lab).mean()
        validation_loss = loss_fn(predicted, torch.FloatTensor(twopoin_lab))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 5 == 0:
        losses.append(loss.item())
        #accur.append(acc.item())
        vali_losses.append(validation_loss.item())

    if i % 10 == 0:
        preds = []
        actuals = []
        for j, val in enumerate(predicted):
            preds.append(val.detach().numpy().tolist())
            actuals.append(twopoin_lab[j])
        rounded = []
        for val in preds:
            temp = []
            for vl in val:
                temp.append(round(vl))
            rounded.append(temp)
        corrects = []
        incorrects = []
        err_corrects = []
        err_incorrects = []
        for k in range(len(rounded)):
            correct = 0
            incorrect = 0
            err_correct = 0
            err_incorrect = 0
            for l in range(len(rounded[0])):
                if rounded[k][l] == actuals[k][l]:
                    correct = correct + 1
                else:
                    incorrect = incorrect + 1
                if actuals[k][l] == 1:
                    if rounded[k][l] == actuals[k][l]:
                        err_correct = err_correct + 1
                    else:
                        err_incorrect = err_incorrect + 1
            corrects.append(correct)
            incorrects.append(incorrect)
            err_corrects.append(err_correct)
            err_incorrects.append(err_incorrect)

        accuracies = []
        err_accuracies = []
        for m in range(len(corrects)):
            temp = (corrects[m] / (corrects[m] + incorrects[m]))
            accuracies.append(temp)
            err_temp = (err_corrects[m] / (err_corrects[m] + err_incorrects[m]))
            err_accuracies.append(err_temp)
        total_err_accuracy.append(mean(err_accuracies))
        total_general_accuracy.append(mean(accuracies))
        print("epoch {}\ttrain_loss : {}\t general accuracy : {}\t error accuracy : {}\ttest_loss {}".format(i, loss, mean(accuracies), mean(err_accuracies), validation_loss))


'''for i in range(len(predicted)):
    print(predicted[i], twopoin_lab[i])'''

#print(losses[0].item())


plt.plot(losses)
plt.title('Train Loss vs Epochs')
plt.xlabel('Epochs (Every 5)')
plt.ylabel('loss')
plt.show()

plt.plot(total_general_accuracy)
plt.title('General Accuracy vs Epochs')
plt.xlabel('Epochs (Every 10)')
plt.ylabel('accuracy')
plt.show()

plt.plot(total_err_accuracy)
plt.title('Error Specific Accuracy vs Epochs')
plt.xlabel('Epochs (Every 10)')
plt.ylabel('accuracy')
plt.show()

plt.plot(vali_losses)
plt.title('Test Loss vs Epochs')
plt.xlabel('Epochs (Every 5)')
plt.ylabel('loss')
plt.show()