from util_functions_train import *
from workspace_utils import active_session

# Main program function defined below
def main():
    
    #Retrieve command line arguments
    in_arg = get_input_args()
    
    # Use gpu if it's available and command line argument for gpu is gpu 
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu=='gpu' else "cpu")

    # Define transforms, datasets and dataloaders
    data_transforms, image_datasets, dataloaders = load_data(in_arg.dir)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    #Build the network
    model, classifier = build_model(in_arg.arch, in_arg.hidden_units)
    
    # Train the network  
    with active_session():
        model_ft = train_model(model.to(device), in_arg.lr, in_arg.epochs, dataloaders, device, dataset_sizes)
    
    # Calculate accuracy on the test set
    acc = calculate_accuracy(model_ft, dataloaders, device)
    
    # Save the checkpoint
    checkpoint_name = "/checkpoint{:.4f}_{}_lr{}_ep{}.pth".format(acc, in_arg.arch, in_arg.lr, in_arg.epochs)
    save_checkpoint(model_ft, in_arg.arch, classifier, image_datasets, in_arg.save_dir+checkpoint_name)

# Call to main function to run the program
if __name__ == "__main__":
    main()