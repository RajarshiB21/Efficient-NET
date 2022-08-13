import torch
from dataset_MNIST import get_data
batch_size = 1000 #Remember to change batch_size here

#Load the Data
train_loader, test_loader = get_data(batch_size)

def get_mean_std(loader):
    ## VAR[X] = E[X**2] - E[X]**2
    #Variance is the expected value of the random variable squared minus the square of the expected value of the random variable
    #STD will be the square root of the variance

    channels_sum, channels_squares_sum, num_batches = 0,0,0

    for data, _ in loader:#We only need data and not targets
        channels_sum+= torch.mean(data, dim=[0,2,3])#The dimensions
        #The first is no.of examples in the batch
        #The dimension one is the number of channels which is not need
        #2 and 3 is the height and width which is required
        channels_squares_sum+=torch.mean(data**2, dim=[0,2,3])
        #Same as above
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches-mean**2)**0.5

    return mean, std


mean, std = get_mean_std(train_loader)
print(mean)
print(std)

##OUTPUT##
#tensor([0.1307])
#tensor([0.3081])