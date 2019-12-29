import torch as t 
import torchvision.transforms as trn
from torch.utils.data import DataLoader

from model import Net
from data import FormDataSet



######## First let's define our transforms and make our datasets

## We won't augment our data

image_tr = { 

    'frames': trn.Compose([

    	trn.Resize(size=(480, 270)),			#### The images size is reduced by 8 to make training faster, and most of all to make my laptop survive to inferences
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])   
    ]),

    'masks': trn.Compose([ 

        trn.Resize(size=(480, 270)),
        trn.ToTensor(),
    ])
   
}


train_data_set = FormDataSet("train_set", image_tr)
valid_data_set = FormDataSet("valid_set", image_tr)

## data loaders

train_data_loader = DataLoader(train_data_set, batch_size=50, shuffle=True)
valid_data_loader = DataLoader(valid_data_set, batch_size=50, shuffle=True)

#### We'll need our data sets sizes to compute the average loss and accuracy

train_size = len(train_data_set)
valid_size = len(valid_data_set)


#### I almost forgot the most important .....

net_model = Net()


########## Now let's define everything we need for training

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')# Here is the loss and optimizer definition
loss_criterion = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net_model.parameters(), lr=0.01)
total_steps = len(train_data_loader)
epochs = 20

   
### Now we're ready to go
start = time.time()
    
#### and initialize what we gonna return 
history = []
best_acc = 0.0

for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
        
    # Set to training mode
    net_model.train()
        
    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0
        
    valid_loss = 0.0
    valid_acc = 0.0
        
    for i, (inputs, masks) in enumerate(train_data_loader):

    	inputs = inputs.to(device)
       	masks =masks.to(device)
           
        # Clean existing gradients
        optimizer.zero_grad()
            
        # Forward pass - compute outputs on input data using the model
        outputs = net_model(inputs)
            
        # Compute loss
        loss = loss_criterion(outputs, masks)
            
        # Backpropagate the gradients
        loss.backward()
            
        # Update the parameters
        optimizer.step()
            
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
            
        # Compute the accuracy
        ret, predictions = t.max(outputs.data, 1)
        correct_counts = predictions.eqmasks.data.view_as(predictions))
            
        # Convert correct_counts to float and then compute the mean
        acc = t.mean(correct_counts.type(t.FloatTensor))
            
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)
            
        

            
    # Validation - No gradient tracking needed
    with t.no_grad():

        # Set to evaluation mode
        net_model.eval()

        # Validation loop
        for j, (inputs, masks) in enumerate(valid_data_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = net_model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, masks)

            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = t.max(outputs.data, 1)
            correct_counts = predictions.eq(masks.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = t.mean(correct_counts.type(t.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)
            
    # Find average training loss and training accuracy
    avg_train_loss = train_loss/train_data_size 
    avg_train_acc = train_acc/train_data_size

    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/valid_data_size

    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
    epoch_end = time.time()
    
    print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s\n\n".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
            
