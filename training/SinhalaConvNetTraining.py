from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy import ndimage
from torchvision import datasets, transforms
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import matplotlib.font_manager as fm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device - ", device)


class Process(object):
    def __call__(self, img):
        convertedImg = img.convert("L")
        invertedImg = ImageOps.invert(convertedImg)
        filteredImg = invertedImg.filter(ImageFilter.MaxFilter(5))
        resizeRatio = 48.0 / max(filteredImg.size)
        newSize = tuple([int(round(x * resizeRatio)) for x in filteredImg.size])
        resizeImg = filteredImg.resize(newSize, Image.LANCZOS)

        resizeImgArray = np.asarray(resizeImg)
        com = ndimage.measurements.center_of_mass(resizeImgArray)
        result = Image.new("L", (64, 64))
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
        result.paste(resizeImg, box)
        return result


transform = transforms.Compose([Process(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainDir = 'D:/Final Year/Final Year Project/dataset-processed1/train'
testDir = 'D:/Final Year/Final Year Project/dataset-processed1/test'

trainingSet = datasets.ImageFolder(trainDir, transform)
print("Full Train Set - ", len(trainingSet))
trainsize = int(round(0.8 * len(trainingSet)))
trainSet, validationSet = torch.utils.data.random_split(trainingSet, [trainsize, len(trainingSet) - trainsize],
                                                        generator=torch.Generator().manual_seed(42))
print("Train Set - ", len(trainSet))
print("Validation Set - ", len(validationSet))
testSet = datasets.ImageFolder(testDir, transform)
print("Test Set - ", len(testSet))
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)

df = pd.read_csv('SinhalaCharacters.csv', header=0)
unicodeList = df["Unicode"].tolist()
char_list = []

for element in unicodeList:
    codeList = element.split()
    charsTogether = ""
    for code in codeList:
        hex = "0x" + code
        charInt = int(hex, 16)
        character = chr(charInt)
        charsTogether += character
    char_list.append(charsTogether)

classes = []
for i in range(31):
    index = int(testSet.classes[i])
    char = char_list[index]
    classes.append(char)

print("Available Classes", classes)


def initializeWeights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 31)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.fc3(x)
        return x


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            , dim=0
        )
    return all_preds


def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(title)
    print(cm)
    prop = fm.FontProperties(fname='Nirmala.ttf')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties=prop)
    plt.yticks(tick_marks, classes, fontproperties=prop)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


net = Net()
net.apply(initializeWeights)
net.to(device)
print("Net - ", net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=0.003, lr=0.001)

x = []
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []
fig, axs = plt.subplots(3)
fig.suptitle('Accuracy')

x_two = []
running_losses = []

for epoch in range(30):
    x.append(epoch)

    curr_train_loss = 0.0
    train_total = 0
    train_correct = 0
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        curr_train_loss += loss.item()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
        running_losses.append(running_loss)
        x_two.append(epoch + i * 64 / len(trainSet))
        running_loss = 0.0

    train_loss.append(curr_train_loss / len(trainSet) * 64)
    train_accuracy.append(100 * train_correct / train_total)

    val_correct = 0
    val_total = 0
    curr_val_loss = 0.0
    with torch.no_grad():
        for data in validationLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            curr_val_loss += criterion(outputs, labels).item()
    val_loss.append(curr_val_loss / len(validationSet) * 64)
    val_accuracy.append(100 * val_correct / val_total)

    print('EPOCH ' + str(epoch + 1))
    print('Training loss: ' + str(train_loss[-1]))
    print('Training accuracy: ' + str(train_accuracy[-1]) + "%")
    print('Validation loss: ' + str(val_loss[-1]))
    print('Validation accuracy: ' + str(val_accuracy[-1]) + "%")
    print('----------------------------------------------------')

    axs[0].plot(x, train_loss, 'r-', val_loss, 'b-')
    axs[1].plot(x, train_accuracy, 'r-', val_accuracy, 'b-')
    axs[2].plot(x_two, running_losses)

plt.show()

dataIteratorTest = iter(testLoader)
data_thing = dataIteratorTest.next()
images, labels = data_thing[0].to(device), data_thing[1].to(device)

imgs = images.cpu()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

outputs = net(images[:10])
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))

lableList = []
predictedList = []
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].to(device), data[1].to(device)
        lableList.extend(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predictedList.extend(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 1240 test images: %f %%' % (100 * correct / total))

with torch.no_grad():
    test_prediction_loader = torch.utils.data.DataLoader(testSet, batch_size=1240)
    test_preds = get_all_preds(net, test_prediction_loader)

cmTesting = confusion_matrix(testSet.targets, test_preds.argmax(dim=1))
plt.figure(figsize=(31, 31))
plot_confusion_matrix(cmTesting, classes, "Confusion Matrix for Test Set")

print("Classification report")
print(classification_report(lableList, predictedList))


def test_class_probabilities(which_class):
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]


def plot_AUC_ROC_curve(classList):
    colors = ['blue', 'darkorange', 'red', 'yellow', 'green', 'pink']
    plt.figure()
    lw = 2

    j = 0
    for i in classList:
        actuals, class_probabilities = test_class_probabilities(i)
        fpr, tpr, _ = roc_curve(actuals, class_probabilities)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[j], lw=lw, label='Class ' + str(i)
                                                         + 'ROC curve (area = %0.2f)' % roc_auc)
        j = j + 1

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + str(classList) + ' classes')
    plt.legend(loc="lower right")
    plt.show()


plot_AUC_ROC_curve([0, 1, 2, 3, 4])
plot_AUC_ROC_curve([5, 6, 7, 8, 9])
plot_AUC_ROC_curve([10, 11, 12, 13, 14])
plot_AUC_ROC_curve([15, 16, 17, 18, 19])
plot_AUC_ROC_curve([20, 21, 22, 23, 24])
plot_AUC_ROC_curve([25, 26, 27, 28, 29, 30])

# torch.save(net.state_dict(), '../Sinhala_conv_net_whiteBG.pt')
