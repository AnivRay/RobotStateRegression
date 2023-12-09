import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from datasets import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def loadData(past_actions_count=3):
    with open("RobotData/actions_3.npy", 'rb') as f:
        savedActions = np.load(f, allow_pickle=True)
        savedActions1 = np.array(savedActions[0][:45000])
        savedActions2 = np.array(savedActions[0][45000:])

    with open("RobotData/observations_3.npy", 'rb') as f:
        savedActions = np.load(f, allow_pickle=True)
        savedObservations1 = np.array(savedActions[0][:45000])
        savedObservations2 = np.array(savedActions[0][45000:])

    X = []
    Y = []
    offset = 20
    for i in range(len(savedActions1) - past_actions_count - offset):
        combinedObsAct1 = np.concatenate([savedObservations1[i:i + past_actions_count], savedActions1[i:i + past_actions_count],], axis=1)
        X.append(combinedObsAct1.flatten())
        Y.append(savedObservations1[i + past_actions_count + offset])

    X = np.array(X)
    Y = np.array(Y)

    trainValSplit = int(0.8 * len(X))
    trainX, valX = X[:trainValSplit], X[trainValSplit:]
    trainY, valY = Y[:trainValSplit], Y[trainValSplit:]

    testX = []
    testY = []
    for i in range(len(savedActions2) - past_actions_count):
        combinedObsAct2 = np.concatenate([savedObservations2[i:i + past_actions_count], savedActions2[i:i + past_actions_count],], axis=1)
        testX.append(combinedObsAct2.flatten())
        testY.append(savedObservations2[i + past_actions_count])

    testX = np.array(testX)
    testY = np.array(testY)
    
    train_dataset = ActionSequenceDataset(trainX, trainY)
    val_dataset = ActionSequenceDataset(valX, valY)
    test_dataset = ActionSequenceDataset(testX, testY)

    return train_dataset, val_dataset, test_dataset

def getModel(modelTag, past_actions_count):

    input_size = past_actions_count * (27 + 8)
    output_size = 27
    numFreqs = 8

    if modelTag == "MLP":
        return MLP(input_size, output_size).to(device)
    elif modelTag == "MLP_Fourier":
        return MLP_Fourier(input_size, output_size, past_actions_count, numFreqs=numFreqs).to(device)
    else: # Is a transformer model
        input_size = 27 + 8
        output_size = 27
        dim_model = 128
        num_heads = 16
        num_layers = 4
        if modelTag == "Transformer":
            return ActionTransformer(input_size, output_size, dim_model, num_heads, num_layers, past_actions_count).to(device)
        elif modelTag == "Transformer_Fourier":
            return ActionTransformerFourier(input_size, output_size, dim_model, num_heads, num_layers, past_actions_count, numFreqs=numFreqs).to(device)
        else:
            print(modelTag, " not implemented yet!")
            return None

def run(trainLoader, valLoader, testLoader, past_actions_count, model="MLP", lr=0.001, epochs=100, outputRoot="output"):
    if not os.path.exists(outputRoot):
        os.mkdir(outputRoot)

    model = getModel(model, past_actions_count)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    minValLoss = None

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_train_batches = 0
        num_val_batches = 0

        for batch_X, batch_Y in trainLoader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            num_train_batches += 1

        for batch_X, batch_Y in valLoader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            epoch_val_loss += loss.item()
            num_val_batches += 1

        if minValLoss is None:
            minValLoss = epoch_val_loss
        elif minValLoss > epoch_val_loss:
            print("Saving model")
            with open(os.path.join(outputRoot, "logfile.txt"), "a") as logFile:
                logFile.write("Saving model\n")
            minValLoss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(outputRoot, "min_vl.pth"))
        
        logString = "Epoch: [%d / %d], Train Loss: %f --- Val Loss: %f" % (epoch + 1, epochs, epoch_train_loss / num_train_batches, epoch_val_loss / num_val_batches)
        print(logString)
        with open(os.path.join(outputRoot, "logfile.txt"), "a") as logFile:
            logFile.write(logString + "\n")

    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(outputRoot, "min_vl.pth")))
        test_loss = 0.0
        num_batches = 0

        for batch_X_test, batch_Y_test in testLoader:
            batch_X_test = batch_X_test.to(device)
            batch_Y_test = batch_Y_test.to(device)
            test_predictions = model(batch_X_test)
            batch_test_loss = criterion(test_predictions, batch_Y_test)
            test_loss += batch_test_loss.item()
            num_batches += 1
        
        logString = "Test Mean Squared Error: " + str(test_loss / num_batches)
        print(logString)
        with open(os.path.join(outputRoot, "logfile.txt"), "a") as logFile:
            logFile.write(logString + "\n")

def main():
    batch_size = 256
    past_actions_count = 1 # Number of previous actions for prediction
    train_dataset, val_dataset, test_dataset = loadData(past_actions_count)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    run(train_dataloader, val_dataloader, test_dataloader, past_actions_count=past_actions_count, model="Transformer_Fourier", outputRoot="outputs_20/Transformer_Fourier_4", epochs=300)


if __name__ == "__main__":
    main()