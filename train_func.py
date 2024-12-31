import torch
import numpy as np

def train_one_epoch(trainloader, model, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in trainloader:
        inputs, labels = batch
        # Move data to the correct device
        inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        # Forward pass
        logits = model(inputs)
        loss = loss_function(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)  # Get the predicted class
        correct_predictions += (preds == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Update total sample count

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(trainloader)
    accuracy = correct_predictions / total_samples
        
    return avg_loss, accuracy


def test_one_epoch(testloader, model, loss_function, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in testloader:
            inputs, labels = batch
            # Move data to the correct device
            inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)
            loss = loss_function(logits, labels)

            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)  # Get the predicted class
            correct_predictions += (preds == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)  # Update total sample count

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(testloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy



def train_one_epoch_with_KD_mix(trainloader, model_upd, loss_function, optimizer, device, alpha=0.5, T=8):
    model_upd.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    # Iterate over the DataLoader for training data
    for batch in trainloader:
        inputs, labels, output_T1, output_T2 = batch
        # Move data to the correct device
        inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
        labels, output_T1, output_T2 = labels.to(device), output_T1.to(device), output_T2.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass
        output_1 = model_upd(inputs)
        
        loss = loss_function(outputs=output_1, labels=labels, outputs_T1=output_T1, outputs_T2=output_T2, alpha=alpha, T=T, device=device)
        
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        
        total_loss += loss.item()
        # Calculate accuracy
        preds = torch.argmax(output_1, dim=1)  # Get the predicted class
        correct_predictions += (preds == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Update total sample count
        
    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(trainloader)
    accuracy = correct_predictions / total_samples
        
    return avg_loss, accuracy


def test_one_epoch_with_KD_mix(testloader, model_upd, loss_function, device, alpha=0.5, T=8):
    model_upd.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    # Iterate over the DataLoader for training data
    for batch in testloader:
        inputs, labels, output_T1, output_T2 = batch
        # Move data to the correct device
        inputs = {key: value.squeeze(0).to(device) for key, value in inputs.items()}
        labels, output_T1, output_T2 = labels.to(device), output_T1.to(device), output_T2.to(device)
        
        # Perform forward pass
        output_1 = model_upd(inputs)
        
        loss = loss_function(outputs=output_1, labels=labels, outputs_T1=output_T1, outputs_T2=output_T2, alpha=alpha, T=T, device=device)
        
        total_loss += loss.item()
        # Calculate accuracy
        preds = torch.argmax(output_1, dim=1)  # Get the predicted class
        correct_predictions += (preds == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Update total sample count
        
    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(testloader)
    accuracy = correct_predictions / total_samples
        
    return avg_loss, accuracy




def train_one_epoch_T5(trainloader, model, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in trainloader:
        inputs, labels = batch
        # Move data to the correct device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)

        # Forward pass
        logits = model(inputs)
        loss = loss_function(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)  # Get the predicted class
        correct_predictions += (preds == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Update total sample count

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(trainloader)
    accuracy = correct_predictions / total_samples
        
    return avg_loss, accuracy


def test_one_epoch_T5(testloader, model, loss_function, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in testloader:
            inputs, labels = batch
            # Move data to the correct device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)
            loss = loss_function(logits, labels)

            total_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)  # Get the predicted class
            correct_predictions += (preds == labels).sum().item()  # Count correct predictions
            total_samples += labels.size(0)  # Update total sample count

    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(testloader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy
