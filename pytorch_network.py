# Neural network implementation using PyTorch
# (Coauthored with GPT-4!)
# M. Eric Carr / mec82@drexel.edu
#=============================================
#import needed packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import csv
#==================================================

#Set input filename here (without .csv)
#This (.csv) should exist, and should be your input (inference or training)
BASENAME='circle_1M'  #(Example: input file is "circle_1M.csv")

#(Associated filenames set automatically from BASENAME)
INPUTFILE=BASENAME+'.csv'
TRAININGFILE=BASENAME+'_training.csv'
BESTFILE=BASENAME+'_best.pth'
LASTFILE=BASENAME+'_last.pth'
IMAGEFILE=BASENAME+'.png'
NETWORKFILE=BASENAME+'_network.csv'
INFERENCEFILE=BASENAME+'_labeled.csv'  #Output file

SAVETIME=60 #Save a checkpoint every x seconds, if not otherwise

#Training / inference / analysis modes
#-------------------------------------
DOTRAINING=1  #0=Don't train -- just create image
              #1=train from random
              #2=load and train
              #3=output network .csv (only)
              #4=draw heatmap of specified node
              #5=load network and run inference on input file, producing output

#Display options
#(Some are not relevant if not producing images)
#-----------------------------------------------
NETWORKANIM=0 #Turn this off if not visualizing network evolution (slow)
EPOCHS_PER_UPDATE = 1
TESTACCURACYEVERY=10 #Test accuracy every N epochs
ZOOMOUT=4.0
XMIN = -0.6*ZOOMOUT
XMAX = 0.6*ZOOMOUT
YMIN = 0.4*ZOOMOUT
YMAX = -0.4*ZOOMOUT
XSTEPS=2400
YSTEPS=1600



#Option to look at a specific node's activation map
#(Only valid for networks that take X/Y coordinate input)
#--------------------------------------------------------
INSPECTLAYER=1
INSPECTNODE=1


#Hardware selection (request CPU or GPU)
#---------------------------------------
USE_GPU=1   #CPU can actually be faster for small networks/batches!
            #0=Use CPU
            #1=Use GPU, if available. Falls back to CPU if not.
            #TODO: Add automatic option based on network size etc?


#ML network parameters
#---------------------
OUTPUT_IS_BINARY=1 #If yes, test accuracy and use BCE error, not MSE
OUTPUT_SIZE=1 #How many output columns in the csv, and how many NN outputs
LABELSFIRST=0 #0=features first in .csv; 1=label(s) first. (may be unreliable?)


#ML model general hyperparameters
#--------------------------------
BATCHSIZE=1024 #was 32 originally
TRAINPCT = 80 # percentage of data for training
MAX_EPOCHS=1000


#Learning rate controls
#(Basic learning rate and dynamic options)
#-----------------------------------------
LEARNRATE=0.001
LEARNDYNAMIC=1 #Learn rate is proportional to bestLoss^learnpower
LEARNFACTOR=1.0 #Dynamic learn rate multiplier (multiplied by VLoss)
LEARNPOWER=1.7 
LEARNMAX=0.005
LEARNMIN=0.00001


#Hidden layer topology.
#(What size/shape is the hidden network?)
# WxD option is default, or custom list can be used
#--------------------------------------------------
HIDDEN_WIDTH=256
HIDDEN_DEPTH=8
HIDDEN_TOPO = [HIDDEN_WIDTH]*HIDDEN_DEPTH
#HIDDEN_TOPO = [1000,500,250,125,100,50,25,20,10,5]
#HIDDEN_TOPO = [4,3,2] #Works for 4-bit primes
#HIDDEN_TOPO = [800]*16
#HIDDEN_TOPO = [200,200,200,200,200,200,200,200]
#HIDDEN_TOPO = [10,10,10,10,10,10]
#HIDDEN_TOPO = [5,5,5,5]
#HIDDEN_TOPO = [10,10,10,10]
#HIDDEN_TOPO = [50]*10
#HIDDEN_TOPO = [500]*6


#==================================================================

# ===============================================
# Please do not modify anything past this point
# (unless you are comfortable experimenting with
# Python and PyTorch scripting, in which case,
# happy hacking!)
#
# (All of the currently-implemented hyperparameters
#  and user options are set above this point.)
# ===============================================

#==================================================================


# Setup code
#--------------------------------------------------------

# Specify the device for computation 
if (USE_GPU) and torch.cuda.is_available():
   #We asked for the GPU, and it's available.
   print("Using the GPU.")
   device = torch.device('cuda')
else:
   if(USE_GPU):
      #We asked to use the GPU, but can't
      print("No GPU available. Using the CPU.")
   else:
      #We didn't ask to use the GPU
      print("CPU selected in settings. Using CPU.")
      if(torch.cuda.is_available()):
         print("(GPU is available)")
      else:
         print("(GPU not available anyway...)")
   device = torch.device('cpu')


class Net(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(Net, self).__init__()
        self.hidden_layers = nn.ModuleList()

        #Create hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                layer = nn.Linear(input_size, hidden_layers[i])
            else:
                layer = nn.Linear(hidden_layers[i - 1], hidden_layers[i])

            #Apply Kaiming initialization
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            self.hidden_layers.append(layer)
        
        # Create output layer (Kaiming initialization)
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity='relu')

    def forward(self, x):
        # Pass through hidden layers with ReLU activation
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))

        # Pass through output layer with sigmoid activation
        x = torch.sigmoid(self.output_layer(x))

        return x

    
# Function to save network structure and weights to CSV
def save_network_to_csv(model, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the first line (input nodes, hidden layers, output nodes)
        input_size = X.shape[1]  # Assuming X is your input tensor
        hidden_layer_sizes = HIDDEN_TOPO
        output_size = OUTPUT_SIZE
        writer.writerow([input_size, len(hidden_layer_sizes), output_size])

        # Write the second line (width of each hidden layer)
        writer.writerow(hidden_layer_sizes)

        # Write the weights and biases of hidden layers
        for layer in model.hidden_layers:
            for bias, weights in zip(layer.bias.data, layer.weight.data):
                writer.writerow([bias.item()] + weights.tolist())

        # Write the weights and biases of output layer
        for bias, weights in zip(model.output_layer.bias.data, model.output_layer.weight.data):
            writer.writerow([bias.item()] + weights.tolist())

    print("Network structure and weights saved to:", filename)

def generate_heatmap(net, layer_index, node_index, x_range, y_range, x_steps, y_steps, device):
    # Generate a grid of values within the specified range
    x_values = np.linspace(x_range[0], x_range[1], x_steps)
    y_values = np.linspace(y_range[0], y_range[1], y_steps)
    X, Y = np.meshgrid(x_values, y_values)
    xy_pairs = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Convert to tensor
    xy_pairs_tensor = torch.tensor(xy_pairs, dtype=torch.float32).to(device)

    # Function to extract the activation of the specified node
    def get_activation(layer, input, output):
        net.activation = output

    # Register hook
    hook = net.hidden_layers[layer_index].register_forward_hook(get_activation)

    # Prepare for storing activation data
    activations = []

    # Evaluate the network in batches (if necessary)
    batch_size = 1024  # Adjust as per your GPU capacity
    num_batches = int(np.ceil(len(xy_pairs_tensor) / batch_size))

    net.eval()
    with torch.no_grad():
        for i in range(num_batches):
            batch = xy_pairs_tensor[i * batch_size:(i + 1) * batch_size]
            _ = net(batch)
            # Extract the activation of the specific node
            batch_activations = net.activation[:, node_index].cpu().numpy()
            activations.append(batch_activations)

    # Remove the hook
    hook.remove()

    # Concatenate and reshape activations to 2D
    activations = np.concatenate(activations).reshape(y_steps, x_steps)

    # Create the binary heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(activations > 0, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='gray')
    plt.colorbar(label='Activation > 0')
    plt.xlabel('Input X')
    plt.ylabel('Input Y')
    plt.title(f'Binary Heatmap of Activation for Node {node_index} in Layer {layer_index}')
    plt.show()


def generate_heatmap_grid(net, hidden_layers, x_range, y_range, x_steps, y_steps, device, figsize_per_subplot=(2, 2)):
    # Function to extract the activation of a specified node
    def get_node_activations(layer_index, node_index):
        x_values = np.linspace(x_range[0], x_range[1], x_steps)
        y_values = np.linspace(y_range[0], y_range[1], y_steps)
        X, Y = np.meshgrid(x_values, y_values)
        xy_pairs = np.stack([X.ravel(), Y.ravel()], axis=1)
        xy_pairs_tensor = torch.tensor(xy_pairs, dtype=torch.float32).to(device)

        def hook_function(module, input, output):
            nonlocal activations
            activations = output.detach()

        hook = net.hidden_layers[layer_index].register_forward_hook(hook_function)

        # Generate activations
        net.eval()
        with torch.no_grad():
            _ = net(xy_pairs_tensor)
        
        # Unregister the hook
        hook.remove()

        # Extract the specific node's activations and reshape
        return activations[:, node_index].cpu().numpy().reshape(y_steps, x_steps)

    # Create subplots
    fig, axes = plt.subplots(nrows=len(hidden_layers), ncols=max(hidden_layers), figsize=(figsize_per_subplot[0]*max(hidden_layers), figsize_per_subplot[1]*len(hidden_layers)))
    axes = axes.flatten() if len(hidden_layers) > 1 else [axes]

    # Generate heatmaps for each node
    node_counter = 0
    for layer_index, num_nodes in enumerate(hidden_layers):
        for node_index in range(num_nodes):
            activations = get_node_activations(layer_index, node_index)
            ax = axes[node_counter]
            ax.imshow(activations > 0, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='gray')
            ax.set_title(f'Layer {layer_index+1} Node {node_index+1}')
            node_counter += 1

        # Disable unused axes
        for _ in range(node_index + 1, max(hidden_layers)):
            axes[node_counter].axis('off')
            node_counter += 1

    plt.tight_layout()
    plt.show()

# Example usage:
# generate_heatmap_grid(net, hidden_layers=HIDDEN_TOPO, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), x_steps=200, y_steps=200, device=device, figsize_per_subplot=(2, 2))



def networkSize(numInput,numHidden,hiddenWidth,numOutput):
    iParams=hiddenWidth*(numInput+1)
    hParams=(numHidden-1)*hiddenWidth*(hiddenWidth+1)
    oParams=numOutput*(hiddenWidth+1)
    return(iParams+hParams+oParams)

def testAccuracy(net, dataloader, device):
    print("Testing accuracy...")
    net.eval()
    correct = 0
    total = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            predicted = (outputs > 0.5).float()  # threshold the outputs to generate binary predictions

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total
    accuracy = accuracy / OUTPUT_SIZE #No 400% accuracy, please!
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'False Positives (Type 1 Error): {false_positives}')
    print(f'False Negatives (Type 2 Error): {false_negatives}')

    return accuracy, false_positives, false_negatives


def generate_rgb_image(net, x_range, y_range, x_steps, y_steps, device, image_file):
    print("Generating RGB image...")

    # Load the best model
    #print("Reloading the best model found...")
    #net.load_state_dict(torch.load(BESTFILE))

    # Switch to evaluation mode
    #print("Switching to evaluation mode...")
    net.eval()

    # Generate x and y values
    x_values = np.linspace(x_range[0], x_range[1], x_steps)
    y_values = np.linspace(y_range[0], y_range[1], y_steps)

    # Create a grid of x, y values
    X, Y = np.meshgrid(x_values, y_values)
    xy_pairs = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Convert to PyTorch tensor             
    xy_pairs_tensor = torch.tensor(xy_pairs, dtype=torch.float32).to(device)

    # Process in batches
    batch_size = 65536  # Adjust based on your GPU memory
    num_batches = int(np.ceil(len(xy_pairs_tensor) / batch_size))

    # Create empty arrays for RGB channels
    RGB = np.empty((len(y_values), len(x_values), 3))

    with torch.no_grad():
        #print("Generating pixel values for RGB channels...")

        for i in range(num_batches):
            #print(f"Batch {i+1} of {num_batches}")
            print('.',end='')
            batch = xy_pairs_tensor[i * batch_size:(i + 1) * batch_size]
            outputs = net(batch)  # Model outputs 3 values per input, for RGB
            if i == 0:
                RGB_flat = outputs
            else:
                RGB_flat = torch.cat((RGB_flat, outputs), 0)
    print('') #EOL
    # Reshape flat outputs to match the image grid, and scale to [0, 1] for image display
    if (OUTPUT_SIZE==3):
        RGB = RGB_flat.cpu().reshape(len(y_values), len(x_values), 3).numpy()
    else:
        RGB = RGB_flat.cpu().reshape(len(y_values), len(x_values), 1).numpy()

    # Ensure RGB values are in the correct range [0, 255] if needed
    RGB = np.clip(RGB, 0, 1)

    # Create the RGB image
    plt.figure(figsize=(x_steps/100, y_steps/100), dpi=100)
    plt.imshow(RGB, extent=(x_range[0], x_range[1], y_range[0], y_range[1]))

    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(image_file, pad_inches=0)

    print(f"RGB image saved as {image_file}")


#---------------------------------
# MAIN PROGRAM
#---------------------------------


startTime=time.time()
lastSavedTime = startTime

bestLoss=999 #Lower is better, and 999 ought to be far worse than any result


# Load data
print("Reading data...")
data = pd.read_csv(INPUTFILE)
print("Total rows in the dataset:", len(data))

if(DOTRAINING != 5):

   # Split into features and target
   print("Splitting data into features and target labels")
   if(LABELSFIRST==1):
       print("Assuming label(s) first.")
       y = data.iloc[:, :-OUTPUT_SIZE]
       X = data.iloc[:, -OUTPUT_SIZE:]
   else:
       print("Assuming features first.")
       X = data.iloc[:, :-OUTPUT_SIZE]
       y = data.iloc[:, -OUTPUT_SIZE:]

   # Standardize the features
   print("Standardizing the features...")
   scaler = StandardScaler()
   X = scaler.fit_transform(X)

   # Convert DataFrame to tensor
   print("Converting DataFrame to tensor...")
   X = torch.tensor(X, dtype=torch.float)
   y = torch.tensor(y.values, dtype=torch.float).view(-1, OUTPUT_SIZE)

   # Create a TensorDataset from your features and labels
   print("Creating dataset...")
   dataset = TensorDataset(X, y)

   # Create a DataLoader for the entire dataset (for testing accuracy)
   full_dataset_dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False)

   # Split the dataset into training and validation datasets
   train_size = int(TRAINPCT/100 * len(dataset))
   val_size = len(dataset) - train_size
   train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

   # Define a DataLoader with a batch size
   print("Defining DataLoader...")
   batch_size = BATCHSIZE  # you can adjust this value depending on your GPU memory size
   train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

   # Define network, loss function and optimizer
   print("Defining network, loss function, and optimizer...")
   net = Net(X.shape[1], HIDDEN_TOPO, OUTPUT_SIZE)
   net = net.to(device)
   if (OUTPUT_IS_BINARY==1):
     criterion = nn.BCELoss()
   else:
      criterion = nn.MSELoss()
   #criterion = nn.L1Loss()
   #optimizer = optim.SGD(net.parameters(), lr=LEARNRATE)
   optimizer = optim.Adam(net.parameters(), lr=LEARNRATE)


if DOTRAINING == 5:

   # Load the network model
   net = Net(input_size=2, hidden_layers=HIDDEN_TOPO, output_size=OUTPUT_SIZE)  # Adjust input_size accordingly
   net.to(device)
   net.load_state_dict(torch.load(BESTFILE))
   net.eval()

   # Load data for inference
   print("Loading data for inference from", INPUTFILE)
   data = pd.read_csv(INPUTFILE)
   scaler = StandardScaler()
   features = scaler.fit_transform(data)

   # Convert to tensor and create DataLoader
   features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
   inference_dataset = TensorDataset(features_tensor)
   inference_loader = DataLoader(inference_dataset, batch_size=BATCHSIZE)

   # Inference
   inferred_labels = []
   with torch.no_grad():
       for batch in inference_loader:
           inputs = batch[0].to(device)
           outputs = net(inputs)
           predicted_labels = (outputs > 0.5).float()  # Binary classification threshold
           inferred_labels.append(predicted_labels.cpu().numpy())
           #print("Processed batch size:", inputs.size(0))

   # Check and handle final batch size
   print("Total batches processed:", len(inferred_labels))
   inferred_labels = np.concatenate(inferred_labels).reshape(-1, 1)

   if inferred_labels.shape[0] != len(data):
       print(f"Warning: Mismatch in data rows ({len(data)}) and labels ({inferred_labels.shape[0]})")

   # Save the inference results
   results = np.hstack((data.values, inferred_labels))
   output_df = pd.DataFrame(results, columns=list(data.columns) + ['Inferred_Label'])
   output_df.to_csv(INFERENCEFILE, index=False)
   print(f"Inference results saved to {INFERENCEFILE}")




if DOTRAINING==2 or DOTRAINING==3 or DOTRAINING==4 :
    # Load the best model
    print("Reloading the best model found...")
    net.load_state_dict(torch.load(BESTFILE))

if DOTRAINING==1 or DOTRAINING==2:
    # Train the network
    print("Training the network...");
    net.train()
    epochs = MAX_EPOCHS
    lastSavedTime=time.time() #Needed to avoid a subtle bug if first epoch too long
    for epoch in range(epochs):
        net.train() #Added b/c progressive images might have it in eval mode
        epochStartTime=time.time()
        for inputs, labels in train_dataloader:  # iterate over batches of data instead of all data
            inputs = inputs.to(device)  # move batch of inputs and labels to device
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epochEndTime=time.time()
            
        #Evaluate the net on the eval data
        net.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)  # move batch of inputs and labels to device
                labels = labels.to(device)

                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_dataloader)
        

            
        # If it's been a long time, save the latest model
        # to a different file (to save work in case we're
        # exploring if deep changes take a long time.)
        if (time.time()-lastSavedTime > SAVETIME):
            torch.save(net.state_dict(), LASTFILE)
            print('Saving the latest state...')
            lastSavedTime=time.time()


        # Print out the progress periodically
        if (epoch+1) % EPOCHS_PER_UPDATE == 0:
           # If we've found a new best network, save it and notify the user
           if val_loss<bestLoss:
               bestLoss = val_loss
               torch.save(net.state_dict(), BESTFILE)
               lastSavedTime=time.time()
               #stepfile=("s6dir_"+str(lastSavedTime)+".pth")
               #torch.save(net.state_dict(), stepfile)
               print('New best! Saved model.')


           curTime=time.time()
           thisLoss=loss.item()
           #print('%.3f Epoch [%d/%d], Loss: %.4f  Best: %.4f' %(curTime-startTime, epoch+1, epochs, thisLoss, bestLoss))
           print('%.3f Epoch [%d/%d], TLoss: %.7f, VLoss: %.7f, BestVL: %.7f' 
                  %(epochEndTime-epochStartTime, epoch+1, epochs, thisLoss, val_loss, bestLoss))
           if LEARNDYNAMIC==1:
              oldLR=LEARNRATE
              #LEARNRATE=val_loss*LEARNFACTOR
              LEARNRATE=(bestLoss**LEARNPOWER)*LEARNFACTOR
              LEARNRATE=min(LEARNRATE,LEARNMAX)
              LEARNRATE=max(LEARNRATE,LEARNMIN)
              if oldLR != LEARNRATE:
                 print('New LR: %.7f' %(LEARNRATE))

               
              
              optimizer = optim.SGD(net.parameters(), lr=LEARNRATE)
           if (NETWORKANIM==1):
               save_network_to_csv(net, NETWORKFILE)

        if(OUTPUT_IS_BINARY and (epoch+1) % TESTACCURACYEVERY == 0):
            accuracy, fp, fn = testAccuracy(net, full_dataset_dataloader, device)
        
if(DOTRAINING==3):

    # Load the best model
    #net.load_state_dict(torch.load(BESTFILE))

    # Save the network structure and weights to CSV
    save_network_to_csv(net, NETWORKFILE)

if(DOTRAINING==4):
    #Create a heatmap for a specific node
    generate_heatmap(net,
                     layer_index=INSPECTLAYER,
                     node_index=INSPECTNODE,
                     x_range=(-1.5, 1.5),
                     y_range=(-1.5, 1.5),
                     x_steps=500,
                     y_steps=500,
                     device=device)



#=============================================

if(DOTRAINING==0 and OUTPUT_SIZE==1):
    #Draw greyscale image
    print("Generating greyscale image...")

    # Load the best model
    print("Reloading the best model found...")
    net.load_state_dict(torch.load(BESTFILE))

    #Switch to evaluation mode
    print("Switching to evaluation mode...")
    net.eval() 

    # Generate x and y values
    x_values = np.linspace(XMIN, XMAX, XSTEPS)
    y_values = np.linspace(YMIN, YMAX, YSTEPS)

    # Create a grid of x, y values
    X, Y = np.meshgrid(x_values, y_values)
    xy_pairs = np.stack([X.ravel(), Y.ravel()], axis=1)  #new

    # Convert to PyTorch tensor             
    xy_pairs_tensor = torch.tensor(xy_pairs, dtype=torch.float32).to(device) #new

    # Process in batches
    batch_size = 65536  # Adjust this based on your GPU memory       #new
    num_batches = int(np.ceil(len(xy_pairs_tensor) / batch_size))   #new


    # Create empty grid for Z values
    Z = np.empty((len(y_values), len(x_values)))

    Z_flat = [] #new


    # Iterate over x and y values
    with torch.no_grad():
        imageStartTime=time.time()
        print("Generating pixel values...")

        #new batch code
        for i in range(num_batches):
            #print("Batch ",i," of ",num_batches)
            print(".",end='')
            batch = xy_pairs_tensor[i * batch_size:(i + 1) * batch_size]
            outputs = net(batch)  # Assuming your model can handle batched inputs
            Z_flat.append(outputs)


    Z_flat = torch.cat(Z_flat)
    Z = Z_flat.cpu().reshape(len(y_values), len(x_values)).numpy()


    # Create the heat map using pcolormesh
    dpi=100
    plt.figure(figsize=(XSTEPS/dpi,YSTEPS/dpi),dpi=dpi)
    #plt.pcolormesh(X, Y, Z, cmap='twilight_shifted', shading='auto')
    #plt.pcolormesh(X, Y, Z, cmap='coolwarm', shading='auto')
    plt.pcolormesh(X, Y, Z, cmap='gray', shading='auto')

    # Adding a colorbar to the plot
    #plt.colorbar(label='Value')

    plt.axis('off')

    # Set labels
    #plt.xlabel('X')
    #plt.ylabel('Y')

    # Set title
    #plt.title('Heatmap of Function')

    #plt.savefig('d:\\heatmap2.png', bbox_inches='tight', pad_inches=0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(IMAGEFILE, pad_inches=0)


    # Show the plot
    #plt.imshow(Z, aspect='equal')
    #plt.show()


if (DOTRAINING==0 and OUTPUT_SIZE==3):
    net.load_state_dict(torch.load(BESTFILE))
    generate_rgb_image(net, x_range=(XMIN, XMAX), y_range=(YMIN, YMAX), x_steps=XSTEPS, y_steps=YSTEPS, device=device, image_file=IMAGEFILE)

    
