import argparse
import torch
from PIL import Image
from torchvision import transforms

def get_input_args():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='ImageClassifier/flowers/test/10/image_07090.jpg', nargs='?', help='Path to the image file')
    parser.add_argument('checkpoint', type=str, default='ImageClassifier/checkpoints/checkpoint.pth', nargs='?',help='Path to the folder where we saved checkpoints')
    
    parser.add_argument('--top_k', type=int, default=5, help='Most likely classes')
    parser.add_argument('--category_names', type=str, default= 'ImageClassifier/cat_to_name.json', help='Mapping of categories to real names:')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU')
    return parser.parse_args()

# Load checkpoint and rebuild the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
######################################################################
# Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image = Image.open(image_path)
  
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  
    image = transform(image)
    return image


##################################################################

# Predict the class from an image file
def predict(image_path, model, topk, mapping, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = torch.FloatTensor(process_image(image_path))
  
    model.eval()
    image = image.unsqueeze(0)
    
    output = model.to(device).forward(image.to(device))
    
    ps = torch.exp(output.cpu())
    
    top_p, top_idx = ps.topk(topk, dim=1)
    
    top_p = top_p.data.numpy()[0]
    top_idx = top_idx.detach().numpy().tolist()[0] 
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_class = [idx_to_class[x] for x in top_idx]
    top_classes_name = [mapping[x] for x in top_class]

    return top_p, top_class, top_classes_name
