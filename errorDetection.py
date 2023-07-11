import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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

'''features = []
for value in jim:
    temp = value[:102]
    features.append(temp)'''

import random

random.Random(4).shuffle(jim)
random.Random(4).shuffle(bob)

tenk_fea = jim[0:3093]
tenk_lab = bob[0:3093]

twopoin_fea = jim[3093:3867]
twopoin_lab = bob[3093:3867]

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
hidden_1_dim = 200
hidden_2_dim = 120
output_dim = 1
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
true_accuracy = []
for i in range(epochs):
    for j, (x_train, y_train) in enumerate(trainloader):
        # calculate output
        output = model(x_train)

        # calculate loss
        loss = loss_fn(output, y_train.reshape(-1, 1))

        # accuracy
        predicted = model(torch.tensor(twopoin_fea, dtype=torch.float32))
        acc = (predicted.reshape(-1).detach().numpy().round() == twopoin_lab).mean()
        validation_loss = loss_fn(predicted, torch.FloatTensor(twopoin_lab))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 5 == 0:
        losses.append(loss.item())
        accur.append(acc.item())
        vali_losses.append(validation_loss.item())

    if i % 10 == 0:
        preds = []
        actuals = []
        for j, val in enumerate(predicted):
            preds.append(val.item())
            actuals.append(twopoin_lab[j])
        rounded = []
        for val in preds:
            rounded.append(round(val))
        correct = 0
        incorrect = 0
        for k in range(len(preds)):
            if float(rounded[k]) == actuals[k][0]:
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        true_accuracy.append(correct / (correct + incorrect))
        print("epoch {}\ttrain_loss : {}\t accuracy : {}\ttest_loss {}".format(i, loss, (correct / (correct + incorrect)),validation_loss))


'''for i in range(len(predicted)):
    print(predicted[i], twopoin_lab[i])'''

#print(losses[0].item())

plt.plot(losses)
plt.title('Train Loss vs Epochs')
plt.xlabel('Epochs (Every 5)')
plt.ylabel('loss')
plt.show()

plt.plot(vali_losses)
plt.title('Validation Loss vs Epochs')
plt.xlabel('Epochs (Every 5)')
plt.ylabel('loss')
plt.show()

plt.plot(true_accuracy)
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs (Every 10)')
plt.ylabel('Accuracy')
plt.show()

