import json
from util_functions_predict import *

# Main program function defined below
def main():
    
    #Retrieve command line arguments
    in_arg = get_input_args()
    
    # mapping from the label number to the actual flower name
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Use gpu if it's available and command line argument for gpu is gpu 
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu=='gpu' else "cpu")
    
    # Load checkpoint and rebuild the model
    model = load_checkpoint(in_arg.checkpoint)
    
    # Process a PIL image for use in the model
    img = process_image(in_arg.input)
    
    # Predict the class from an image file
    probs, classes, classes_name = predict(in_arg.input, model, in_arg.top_k, cat_to_name, device)
    print("Probability:", probs)
    print("Classes:",classes)
    print("Classes Name:",classes_name)

# Call to main function to run the program
if __name__ == "__main__":
    main()