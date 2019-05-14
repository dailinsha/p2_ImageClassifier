import argparse
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import PIL
from torch.autograd import Variable
from train import load_pretrained_model
import json
import torch

def set_args():
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument('--input_dir', default='ImageClassifier/flowers/test/10/image_07090.jpg', type=str, help='image to process and predict')
    parser.add_argument('--checkpoint_dir', default='checkpoint.pth', type=str, help='checkpoint directory')  
    parser.add_argument('--json_dir', default='ImageClassifier/cat_to_name.json', type=str, help=' mapping the category to name') 
    parser.add_argument('--topk', default=3, type=int, help='top k value')
    parser.add_argument('--gpu', default= 'gpu', type=str, help='devices, options: gpu, cpu')  
    
    return parser.parse_args() 

def load_model(checkpoint_dir):
    checkpoint = torch.load(checkpoint_dir)
    model, ins = load_pretrained_model(checkpoint['pretrain_model'])
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']  
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    criterion = checkpoint['criterion']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    return model, optimizer, criterion, epochs

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = PIL.Image.open(image)
    size = np.max(pil_image.size)*256/np.min(pil_image.size)
    pil_image.thumbnail((size,size))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image)/225
    
    means = np.array([0.485,0.456,0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means)/std
    
    np_image = np_image.transpose(2,0,1)
    torth_image = torch.from_numpy(np_image)
    return torth_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)    
    return ax

def predict(image, model, gpu, json_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
     
    image = Variable(image.unsqueeze(0).float())
    model.eval()
     
    if torch.cuda.is_available() and gpu == 'gpu':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")        
        
#     if gpu =='gpu' and torch.cuda.is_available():
#         model.cuda()
#         image = image.cuda()
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        predict = model.forward(image)
        results = torch.exp(predict).topk(topk, dim=1)
    
    probs = results[0][0].cpu().numpy()
    ids = results[1][0].cpu().numpy() 
    class_ids =  model.class_to_idx
    id_classes = {}
    names = []
    for x in class_ids:
        id_classes[class_ids[x]] = x  
    classes = [id_classes[x] for x in ids.tolist()]
    
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    for c in classes:
        names.append(cat_to_name[c])
    
    return probs, ids, classes, names

def plot_results(probs, names, ax = None):    
    print('Predict categories:', names) 
    print('Predict probilities:', probs)
    if ax is None:
        fig, ax = plt.subplots()
    names_df = pd.DataFrame(names)
    probs_df = pd.DataFrame(probs)
    name_prob = pd.concat([names_df, probs_df],axis =1)
    name_prob.columns = ['names', 'probs']
    name_prob.sort_values('probs', ascending = True, inplace = True)
    name_prob['probs'].plot.barh(ax =ax)
    plt.yticks(range(len(name_prob)),name_prob['names'])
    plt.show()
    return ax
    
def main():
    args = set_args()
    print(args)
    model, optimizer, criterion, epochs = load_model(args.checkpoint_dir)
    image = process_image(args.input_dir)
#     imshow(image)
    probs, ids, classes, names = predict(image, model, args.gpu, args.json_dir, args.topk)
    print(names)
    print(classes)
    print(probs)
#    plot_results(probs, names)
    pass

if __name__ == '__main__':
    main()
