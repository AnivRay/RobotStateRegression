import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plotBarGraph(title, outFolders, catNames=None):
    testErrors = []
    percentErrors = []

    for outFolder in outFolders:
        with open(os.path.join(outFolder, "logfile.txt"), 'r') as inFile:
            lines = inFile.readlines()
        testError = float(lines[-1].split()[-1])
        testErrors.append(testError)
        percentErrors.append((testError - testErrors[0]) / testErrors[0] * 100)

    print(percentErrors)
    plt.figure(figsize=(10, 6))
    labels = catNames if catNames is not None else [os.path.split(outFolder)[-1] for outFolder in outFolders]
    sns.barplot(x=labels, y=percentErrors)
    plt.axhline(0, color='black')
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel('Percent Reduction Test Error')
    fileName = '_'.join(title.split())
    plt.savefig("{}.png".format(fileName))
    plt.clf()

outFolders = ["outputs_sequential_features/MLP", "outputs_final/MLP_Fourier", "outputs_sequential_features/Transformer", "outputs_final/Transformer_Fourier"]
plotBarGraph("Test Error by Model", outFolders)

outFolders = ["outputs_sequential_features/MLP_Fourier", "outputs_final/MLP_Fourier", "outputs_sequential_features/Transformer_Fourier", "outputs_final/Transformer_Fourier"]
plotBarGraph("Test Error by Fourier Frequency Type", outFolders, catNames=["MLP_Sequential", "MLP_Random", "Transformer_Sequential", "Transformer_Random"])

