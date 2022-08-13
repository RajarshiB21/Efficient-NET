#Importing the libraries
import torch
import torchmetrics

##Calculating Metrics
def calc_metrics(targets, predictions):
    #We will be using the pytorch metrics library to compute the precision, recall and f1_score
    #The functions take the predictions and the targets, with various other parameters
    #multiclass = True since we have multiple classes in our dataset

    #Precision
    #TP / TP + FP
    #'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).

    precision = torchmetrics.functional.precision(predictions, targets, num_classes=10, average= 'macro', multiclass = True )

    #Recall
    #TP / TP + FN
    #'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    recall = torchmetrics.functional.recall(predictions, targets, num_classes=10, average= 'macro', multiclass = True)

    #F1 Score
    #Threshold for transforming probability or logit predictions to binary
    #'macro': Calculate the metric for each class separately, and average the metrics across classes (with equal weights for each class).
    f1_score = torchmetrics.functional.f1_score(predictions, targets, threshold=0.5, num_classes=10, average= 'macro', multiclass = True)
    return precision, recall, f1_score


#Check accuracy on training and testing
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    if loader.dataset.train:
        print("Checking accuracy on training")
    else:
        print("Checking accuracy on testing")

    with torch.no_grad():
        #we dont need gradients for calculating acccuracy
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)
            #We already flatten it in the cnn

            scores = model(x)
            #Shape of scores is 64x10
            #We want to know which one is the max of those 10 digits

            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct}/{num_samples} with accuracy {num_correct/num_samples * 100:.2f}")

    model.train()
    return num_correct/num_samples * 100
