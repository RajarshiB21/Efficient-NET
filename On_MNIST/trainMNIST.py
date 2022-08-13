import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from model import *
from dataset_MNIST import get_data
from torch.utils.tensorboard import SummaryWriter
from metrics import check_accuracy, calc_metrics

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
#input_size = 784
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 1000
num_epochs = 25
version = "b0"

#Load the data
train_loader, test_loader = get_data(batch_size)

#initialise the network
version = version
phi, res, drop_rate = phi_values[version]
model = EfficientNet(
    version=version,
    num_classes=num_classes,
    in_channels=in_channels,
).to(device)

#Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(f'runs/MNIST/ENET_B0_METRICS')

#Train network
step = 0
for epoch in range(num_epochs):#Goes over each epoch
    #initialise tqdm in separately
    loop = tqdm(enumerate(train_loader), total=len(train_loader))#add leave=False  to keep it on a single line
    for batch_idx, (data, targets) in loop:
        #Goes over every item in the batch
        #which batch index do we have is found from above
        data = data.to(device=device)
        targets = targets.to(device=device)
        #Output for data.shape is 64,1,28,28
        #We want it to be 64,784

        #data = data.reshape(data.shape[0], -1)
        # We already flatten it in the cnn

        #print(data.shape)

        #Forward part
        scores = model(data)
        loss = criterion(scores, targets)
        #print(targets)

        #Backward part
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent or Adam Step
        optimizer.step()
        #update weights computed in loss.backward()

        #Calculate running training acuracy
        _, predictions = scores.max(1)
        num_correct = (predictions==targets).sum()
        running_train_acc = float(num_correct)/float(data.shape[0])

        #Calculate Metrics
        precision, recall, f1_score = calc_metrics(targets, predictions)

        #Update tqdm
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(), acc=running_train_acc, precision=precision.item(), recall=recall.item(), f1_score = f1_score.item())

        # Update Tensorboard
        writer.add_scalar('Training Loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)
        writer.add_scalar('Training Precision', precision, global_step=step)
        writer.add_scalar('Training Recall', recall, global_step=step)
        writer.add_scalar('Training f1_score', f1_score, global_step=step)
        step += 1



train_acc = check_accuracy(train_loader,model,device)
test_acc = check_accuracy(test_loader,model,device)
