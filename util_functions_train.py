import torchvision
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

from collections import OrderedDict
import time
import copy
import argparse

densenet161 = models.densenet161(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)                    
    
three_models = {'densenet161': {'model_name': densenet161, 'inputs': 2208}, 
                'alexnet': {'model_name': alexnet, 'inputs': 9216}, 
                'vgg16': {'model_name': vgg16, 'inputs': 25088}} 

def get_input_args():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default='ImageClassifier/flowers', nargs='?', help='Path to the flowers image files')
    parser.add_argument('--save_dir', type=str, default='ImageClassifier/checkpoints', help='Path to the folder where we save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet161', help='CNN model architecture')
    parser.add_argument('--lr', type=float, default=0.001, help='Coefficient that scale delta before it is applied to the parameters')
    parser.add_argument('--hidden_units', type=int, default=2000, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU')
    return parser.parse_args()
#############################################################################################

def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    size = 128
    # Define transforms for the training, validation, and testing sets
    data_transforms    = {
        'train' : transforms.Compose([
            transforms.RandomRotation(55),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(train_dir, transform=data_transforms['test'])
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=size, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=size, shuffle=True),
    }
    
    return data_transforms, image_datasets, dataloaders
##########################################################################################################################

def build_model(arch, hidden_size):
    # Build the network
                                          
    input_size = three_models[arch]['inputs']
    model = three_models[arch]['model_name']
    output_size = 102
    dropout = 0.4

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_size)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(p=dropout)),
                              ('fc2', nn.Linear(hidden_size, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    return model, classifier 
##########################################################################################################################

def train_model(model, lr, num_epochs, dataloaders, device, dataset_sizes):
    since = time.time()
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

##########################################################################################################################

def calculate_accuracy(model, dataloaders, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

    print('Accuracy of the network on the test images: %d %%' % (acc))
    return acc
    
##########################################################################################################################
def save_checkpoint(model, arch, classifier, image_datasets, path):
    model = model.cpu()
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'input_size': three_models[arch]['inputs'],
                  'output_size': 120,
                  'model': three_models[arch]['model_name'],
                  'classifier': classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
             }
                        
    torch.save(checkpoint, path) 
    
############################################################################
