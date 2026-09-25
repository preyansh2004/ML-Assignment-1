import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Finding the main project folder
projectFolder = Path(__file__).resolve().parent.parent

# Creating a folder where images will be saved
plotFolder = projectFolder / "images"
plotFolder.mkdir(exist_ok=True)


# Creating a function for simultaneous diagonalization
def simultaneousDiagonalization(fileName):
    print("DATASET:", fileName)

    # Reading the dataset
    filePath = projectFolder / "csv" / fileName
    data = pd.read_csv(filePath)

    # Getting the two classes
    class0 = data[data["label"] == 0][["x1", "x2"]]
    class1 = data[data["label"] == 1][["x1", "x2"]]

    # Converting to numpy arrays for transformations
    points0 = class0.to_numpy()
    points1 = class1.to_numpy()

    # Printing dataset shape and class counts
    print("\nDataset shape:",data.shape)
    print("\nClass counts:",data["label"].value_counts())

    # Calculating original mean vectors
    meanClass0 = class0.mean()
    meanClass1 = class1.mean()

    # Printing original mean vectors
    print("\nORIGINAL MEAN VECTORS")
    print("\nMean vector for Class 0:")
    print(meanClass0.to_numpy())
    print("\nMean vector for Class 1:")
    print(meanClass1.to_numpy())

    # Calculating original covariance matrices
    covClass0 = np.cov(class0, rowvar=False, bias=True)
    covClass1 = np.cov(class1, rowvar=False, bias=True)

    # Printing original covariance matrices
    print("\nORIGINAL COVARIANCE MATRICES")
    print("\nCovariance matrix for Class 0:")
    print(covClass0)
    print("\nCovariance matrix for Class 1:")
    print(covClass1)

    # Calculating eigen decomposition for both classes
    eigenvalues0, eigenvectors0 = np.linalg.eigh(covClass0)
    eigenvalues1, eigenvectors1 = np.linalg.eigh(covClass1)

    # Printing eigenvalues and eigenvectors for both classes
    print("\nEIGEN DECOMPOSITION")
    print("\nEigenvalues for Class 0:")
    print(eigenvalues0)
    print("\nEigenvectors for Class 0:")
    print(eigenvectors0)
    print("\nEigenvalues for Class 1:")
    print(eigenvalues1)
    print("\nEigenvectors for Class 1:")
    print(eigenvectors1)

    # First transformation
    # Using eigenvectors of Class 0
    transformed0 = points0 @ eigenvectors0
    transformed1 = points1 @ eigenvectors0

    print("\nAFTER FIRST TRANSFORMATION")

    meanTransformed0 = transformed0.mean(axis=0)
    meanTransformed1 = transformed1.mean(axis=0)
    covTransformed0 = np.cov(transformed0,rowvar=False,bias=True)
    covTransformed1 = np.cov(transformed1,rowvar=False,bias=True)

    # Printing mean vectors and covariance matrices after first transformation
    print("\nMean vector for Class 0:")
    print(meanTransformed0)
    print("\nMean vector for Class 1:")
    print(meanTransformed1)
    print("\nCovariance matrix for Class 0:")
    print(covTransformed0)
    print("\nCovariance matrix for Class 1:")
    print(covTransformed1)

    # Whitening of matrix
    whiteningMatrix = np.diag(1 / np.sqrt(eigenvalues0))
    whitened0 = transformed0 @ whiteningMatrix
    whitened1 = transformed1 @ whiteningMatrix

    # Printing mean vectors and covariance matrices after whitening
    print("\nAFTER WHITENING")
    print("\nWhitening matrix:")
    print(whiteningMatrix)

    meanWhitened0 = whitened0.mean(axis=0)
    meanWhitened1 = whitened1.mean(axis=0)
    covWhitened0 = np.cov(whitened0,rowvar=False,bias=True)
    covWhitened1 = np.cov(whitened1,rowvar=False,bias=True)

    # Printing mean vectors and covariance matrices after whitening
    print("\nMean vector for Class 0:")
    print(meanWhitened0)
    print("\nMean vector for Class 1:")
    print(meanWhitened1)
    print("\nCovariance matrix for Class 0:")
    print(covWhitened0)
    print("\nCovariance matrix for Class 1:")
    print(covWhitened1)

    # Creating Eigen decomposition of Class 1
    # after whitening
    eigenvaluesFinal, eigenvectorsFinal = np.linalg.eigh(covWhitened1)

    # Printing final eigenvalues and eigenvectors
    print("\nFINAL EIGEN DECOMPOSITION")
    print("\nFinal eigenvalues:")
    print(eigenvaluesFinal)
    print("\nFinal eigenvectors:")
    print(eigenvectorsFinal)

    # Final transformation
    final0 = whitened0 @ eigenvectorsFinal
    final1 = whitened1 @ eigenvectorsFinal

    # Ploting final diagonalized data only for twogaussians datasets
    if fileName in ["twogaussians33.csv", "twogaussians42.csv"]:
        plt.figure(figsize=(7, 6))
        plt.scatter(final0[:, 0],final0[:, 1],marker="o",label="Class 0")
        plt.scatter(final1[:, 0],final1[:, 1],marker="^",label="Class 1")
        plt.xlabel("Transformed Feature 1")
        plt.ylabel("Transformed Feature 2")
        plt.title(f"{fileName} - Final Diagonalized Data")
        plt.legend()
        plt.grid(True)

        outputFile = plotFolder / f"{fileName.replace('.csv', '')}_diagonalized.png"
        plt.savefig(outputFile, dpi=300, bbox_inches="tight")
        plt.close()

        # Printing the path of the saved plot
        print(f"Final diagonalized plot saved to: {outputFile}")

    print("\nAFTER FINAL TRANSFORMATION")
    meanFinal0 = final0.mean(axis=0)
    meanFinal1 = final1.mean(axis=0)
    covFinal0 = np.cov(final0,rowvar=False,bias=True)
    covFinal1 = np.cov(final1,rowvar=False,bias=True)

    # Printing mean vectors and covariance matrices after final transformation
    print("\nMean vector for Class 0:")
    print(meanFinal0)
    print("\nMean vector for Class 1:")
    print(meanFinal1)
    print("\nCovariance matrix for Class 0:")
    print(covFinal0)
    print("\nCovariance matrix for Class 1:")
    print(covFinal1)

    # Creating visualization for transformations
    fig, axes = plt.subplots(2,2,figsize=(12, 10))

    # Original
    axes[0, 0].scatter(points0[:, 0],points0[:, 1],label="Class 0",marker="o",alpha=0.6)
    axes[0, 0].scatter(points1[:, 0],points1[:, 1],label="Class 1",marker="^",alpha=0.6)
    axes[0, 0].set_title("Original Data")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # First transformation
    axes[0, 1].scatter(transformed0[:, 0],transformed0[:, 1],label="Class 0",marker="o",alpha=0.6)
    axes[0, 1].scatter(transformed1[:, 0],transformed1[:, 1],label="Class 1",marker="^",alpha=0.6)
    axes[0, 1].set_title("After First Transformation")
    axes[0, 1].set_xlabel("x1'")
    axes[0, 1].set_ylabel("x2'")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Whitening
    axes[1, 0].scatter(whitened0[:, 0],whitened0[:, 1],label="Class 0",marker="o",alpha=0.6)
    axes[1, 0].scatter(whitened1[:, 0],whitened1[:, 1],label="Class 1",marker="^",alpha=0.6)
    axes[1, 0].set_title("After Whitening")
    axes[1, 0].set_xlabel("x1''")
    axes[1, 0].set_ylabel("x2''")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Final
    axes[1, 1].scatter(final0[:, 0],final0[:, 1],label="Class 0",marker="o",alpha=0.6)
    axes[1, 1].scatter(final1[:, 0],final1[:, 1],label="Class 1",marker="^",alpha=0.6)
    axes[1, 1].set_title("After Final Transformation")
    axes[1, 1].set_xlabel("x1'''")
    axes[1, 1].set_ylabel("x2'''")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Adding dataset name to the overall figure
    fig.suptitle(fileName.replace(".csv", ""),fontsize=16)
    plt.tight_layout()

    # Saving the images
    plotName = fileName.replace(".csv","_transformations.png")
    plotFile = plotFolder / plotName
    plt.savefig(plotFile,dpi=300,bbox_inches="tight")
    plt.close()

    # Printing the path of the saved plot
    print("\nPlot saved to:")
    print(plotFile)

# Running the function on all 3 datasets
datasets = ["twogaussians33.csv","twogaussians42.csv","moons1.csv"]
for dataset in datasets:
    simultaneousDiagonalization(dataset)