import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import sys
from collections import OrderedDict

def set_args():
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument('--data_dir', default='ImageClassifier/flowers', type=str, help='raw data directory')
    parser.add_argument('--pretrained_model', default='densenet', help='pretrained model architectures, options: vgg, densenet, alexnet')
    parser.add_argument('--hidden_units_clf', default=[512], type=list, help='default hidden layer architecture for my own classification')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='default learning rate' )    
    parser.add_argument('--epochs', default=10, type=int, help='default training epochs')
    parser.add_argument('--gpu', default= 'gpu', type=str, help='devices, options: gpu, cpu')       
    return parser.parse_args()  

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([ transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(valid_dir, data_transforms['val']),
        'test': datasets.ImageFolder(test_dir, data_transforms['test'])    
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']
    }
    class_names = image_datasets['train'].classes
    return image_datasets, dataloaders
   
def load_pretrained_model(model_name):
    """
    Parameters:
        model_name - Pretrained CNN model name from pytorchversion.models choose from 'vgg', 'densenet', 'alexnet'
    Returns:
        model - Pretrained feature CNN with untrained classifier
        ins - Input size for the classifer layer
    """
    if model_name == 'vgg':        
        model = models.vgg16(pretrained=True)
        ins = 25088
        print('Loading the model vgg...')
    elif model_name == 'densenet':
        model = models.densenet121(pretrained=True)
        ins = 1024
        print('Loading the model densenet...')
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        ins = 9216
        print('Loading the model alexnet...')
    else:
        print('Model not recognized')
        sys.exit
    print('Model load successed.')
    return model, ins
    
def build_clf(model, ins, layer_list, outs=102,  drop_p = 0.5):
    """
    Parameters:
        model - Pretrained model
        ins - input size
        layers - hidden layer list with numbers of each layers []
        outs- output size (default - 102)
        learning_rate - determines the learning rate for the optimizer
        drop_p - determines the dropout probability for the classifier(default- 0.5)
    Returns:
        model - new model
    """
    for param in model.parameters():
        param.requires_grad = False
    net_layers = OrderedDict()
    
    for i in range(len(layer_list)+1):
        if i ==0:
            h1 = ins
        if i < len(layer_list):
            h2 = layer_list[i]
            net_layers.update({'fc{}'.format(i): nn.Linear(h1, h2)})
            net_layers.update({'relu{}'.format(i):nn.ReLU()})
            net_layers.update({'drop{}'.format(i):nn.Dropout(drop_p)})
            h1 = h2
        else:
            h2 = outs
            net_layers.update({'fc{}'.format(i):nn.Linear(h1, h2)})
            net_layers.update({'output':nn.LogSoftmax(dim=1)})
    
    clf= nn.Sequential(net_layers)
    model.classifier = clf
    return model 

def train_model(model, criterion, optimizer, epochs, gpu, dataloaders):    
    """
    Parameters:
        model - reconstructed model with our own classifier
        criterion - chosen criterion (default - nn.NLLLoss())
        optimizer - chosen optimizer (default - nn.Adam())
        epochs - number of epochs
        device - running envinerment cpu/cuda (default - cuda)
    Returns:
        model - trained  model
        printout the loss and accurancy for each epoch
    """
    if torch.cuda.is_available() and gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    model.to(device)
#     best_accuracy = float("inf")
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)                
            optimizer.zero_grad()        
            logps = model(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()        
            running_loss +=loss.item()
        else:
            valid_loss = 0
            accuracy = 0        
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#             if accuracy < best_accuracy:
#                 best_accuracy = accuracy
#                 best_epoch = epoch+1
            print("Training Loss: {:.3f}.. ".format(running_loss/len(dataloaders['train'])),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['val'])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))
    
    return model, accuracy/len(dataloaders['val'])

def save_model(image_datasets, model, optimizer, criterion, epochs):
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = { 
        'pretrain_model': 'densenet',
        'classifier': model.classifier,
        'optimizer' : optimizer,
        'criterion' : criterion,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict() ,
        'epochs':epochs,
        'class_to_idx': model.class_to_idx
        }
    torch.save(checkpoint, 'checkpoint.pth')
    pass

def main():
    args = set_args()
    print(args)
    image_datasets, dataloaders = load_data(args.data_dir)
    model, ins = load_pretrained_model(args.pretrained_model)    
    model = build_clf(model, ins, args.hidden_units_clf)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    model, accuracy = train_model(model, criterion, optimizer, args.epochs, args.gpu, dataloaders)
    save_model(image_datasets, model, optimizer, criterion, args.epochs)
    pass

if __name__ == '__main__':
    main()