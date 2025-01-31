import torch
import torch.nn as nn
import torch.optim as optim
#from networks.StartingNetwork import StartingNetwork
import tensorflow as tf
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    step = 0
    for epoch in range(epochs):
        print("Epoch {epoch + 1} of {epochs}", end='\r')

        # Loop over each batch in the dataset
        for batchx in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            images, labels = batchx
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = outputs.argmax(axis = 1)
            # Periodically evaluate our model + log to Tensorboard

            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                # above loss above
                accuracy = compute_accuracy(outputs, labels)
                print(accuracy)
                # tf.summary.scalar('accuracy', accuracy, step=epoch)
                
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                
                # with torch.no_grad():
                #     model.eval()
                #     evaluate(val_loader, model, loss_fn)
                #     model.train()
                
            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    print(outputs, labels)
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(axis = 1)
        #print("shapes:")
        #print(outputs.shape, labels.shape)
        #acc = compute_accuracy(predictions, labels)
        # loss??
        correct += (labels == predictions).int().sum()
        total += len(predictions)
    print('Accuracy:', (correct / total).item())
    
