import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Find the main project folder
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent

# Folder where plots will be saved
plot_folder = project_folder / "plots"
plot_folder.mkdir(exist_ok=True)


# --------------------------------------------------
# Function for simultaneous diagonalization
# --------------------------------------------------

def simultaneous_diagonalization(file_name):

    print("\n")
    print("=" * 70)
    print("DATASET:", file_name)
    print("=" * 70)

    # --------------------------------------------------
    # Step 1: Read the dataset
    # --------------------------------------------------

    file_path = project_folder / "csv" / file_name

    data = pd.read_csv(file_path)

    # Get the two classes
    class_0 = data[data["label"] == 0][["x1", "x2"]]
    class_1 = data[data["label"] == 1][["x1", "x2"]]

    # Convert to NumPy arrays for transformations
    points_0 = class_0.to_numpy()
    points_1 = class_1.to_numpy()

    print("\nDataset shape:")
    print(data.shape)

    print("\nClass counts:")
    print(data["label"].value_counts())


    # --------------------------------------------------
    # Step 2: Calculate original mean vectors
    # --------------------------------------------------

    mean_class_0 = class_0.mean()
    mean_class_1 = class_1.mean()

    print("\n--- ORIGINAL MEAN VECTORS ---")

    print("\nMean vector for Class 0:")
    print(mean_class_0.to_numpy())

    print("\nMean vector for Class 1:")
    print(mean_class_1.to_numpy())


    # --------------------------------------------------
    # Step 3: Calculate original covariance matrices
    # --------------------------------------------------

    cov_class_0 = np.cov(class_0, rowvar=False, bias=True)
    cov_class_1 = np.cov(class_1, rowvar=False, bias=True)

    print("\n--- ORIGINAL COVARIANCE MATRICES ---")

    print("\nCovariance matrix for Class 0:")
    print(cov_class_0)

    print("\nCovariance matrix for Class 1:")
    print(cov_class_1)


    # --------------------------------------------------
    # Step 4: Eigen decomposition
    # --------------------------------------------------

    eigenvalues_0, eigenvectors_0 = np.linalg.eigh(cov_class_0)
    eigenvalues_1, eigenvectors_1 = np.linalg.eigh(cov_class_1)

    print("\n--- EIGEN DECOMPOSITION ---")

    print("\nEigenvalues for Class 0:")
    print(eigenvalues_0)

    print("\nEigenvectors for Class 0:")
    print(eigenvectors_0)

    print("\nEigenvalues for Class 1:")
    print(eigenvalues_1)

    print("\nEigenvectors for Class 1:")
    print(eigenvectors_1)


    # --------------------------------------------------
    # Step 5: First transformation
    # Use eigenvectors of Class 0
    # --------------------------------------------------

    transformed_0 = points_0 @ eigenvectors_0
    transformed_1 = points_1 @ eigenvectors_0

    print("\n--- AFTER FIRST TRANSFORMATION ---")

    mean_transformed_0 = transformed_0.mean(axis=0)
    mean_transformed_1 = transformed_1.mean(axis=0)

    cov_transformed_0 = np.cov(
        transformed_0,
        rowvar=False,
        bias=True
    )

    cov_transformed_1 = np.cov(
        transformed_1,
        rowvar=False,
        bias=True
    )

    print("\nMean vector for Class 0:")
    print(mean_transformed_0)

    print("\nMean vector for Class 1:")
    print(mean_transformed_1)

    print("\nCovariance matrix for Class 0:")
    print(cov_transformed_0)

    print("\nCovariance matrix for Class 1:")
    print(cov_transformed_1)


    # --------------------------------------------------
    # Step 6: Whitening
    # --------------------------------------------------

    whitening_matrix = np.diag(
        1 / np.sqrt(eigenvalues_0)
    )

    whitened_0 = transformed_0 @ whitening_matrix
    whitened_1 = transformed_1 @ whitening_matrix

    print("\n--- AFTER WHITENING ---")

    print("\nWhitening matrix:")
    print(whitening_matrix)

    mean_whitened_0 = whitened_0.mean(axis=0)
    mean_whitened_1 = whitened_1.mean(axis=0)

    cov_whitened_0 = np.cov(
        whitened_0,
        rowvar=False,
        bias=True
    )

    cov_whitened_1 = np.cov(
        whitened_1,
        rowvar=False,
        bias=True
    )

    print("\nMean vector for Class 0:")
    print(mean_whitened_0)

    print("\nMean vector for Class 1:")
    print(mean_whitened_1)

    print("\nCovariance matrix for Class 0:")
    print(cov_whitened_0)

    print("\nCovariance matrix for Class 1:")
    print(cov_whitened_1)


    # --------------------------------------------------
    # Step 7: Eigen decomposition of Class 1
    # after whitening
    # --------------------------------------------------

    eigenvalues_final, eigenvectors_final = np.linalg.eigh(
        cov_whitened_1
    )

    print("\n--- FINAL EIGEN DECOMPOSITION ---")

    print("\nFinal eigenvalues:")
    print(eigenvalues_final)

    print("\nFinal eigenvectors:")
    print(eigenvectors_final)


    # --------------------------------------------------
    # Step 8: Final transformation
    # --------------------------------------------------

    final_0 = whitened_0 @ eigenvectors_final
    final_1 = whitened_1 @ eigenvectors_final

    # Plot final diagonalized data only for twogaussians datasets
    if file_name in ["twogaussians33.csv", "twogaussians42.csv"]:

        plt.figure(figsize=(7, 6))

        plt.scatter(
            final_0[:, 0],
            final_0[:, 1],
            marker="o",
            label="Class 0"
        )

        plt.scatter(
            final_1[:, 0],
            final_1[:, 1],
            marker="^",
            label="Class 1"
        )

        plt.xlabel("Transformed Feature 1")
        plt.ylabel("Transformed Feature 2")
        plt.title(f"{file_name} - Final Diagonalized Data")
        plt.legend()
        plt.grid(True)

        output_file = plot_folder / f"{file_name.replace('.csv', '')}_diagonalized.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Final diagonalized plot saved to: {output_file}")

    print("\n--- AFTER FINAL TRANSFORMATION ---")

    mean_final_0 = final_0.mean(axis=0)
    mean_final_1 = final_1.mean(axis=0)

    cov_final_0 = np.cov(
        final_0,
        rowvar=False,
        bias=True
    )

    cov_final_1 = np.cov(
        final_1,
        rowvar=False,
        bias=True
    )

    print("\nMean vector for Class 0:")
    print(mean_final_0)

    print("\nMean vector for Class 1:")
    print(mean_final_1)

    print("\nCovariance matrix for Class 0:")
    print(cov_final_0)

    print("\nCovariance matrix for Class 1:")
    print(cov_final_1)


    # --------------------------------------------------
    # Step 9: Create visualization
    # --------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10)
    )

    # Original
    axes[0, 0].scatter(
        points_0[:, 0],
        points_0[:, 1],
        label="Class 0",
        marker="o",
        alpha=0.6
    )

    axes[0, 0].scatter(
        points_1[:, 0],
        points_1[:, 1],
        label="Class 1",
        marker="^",
        alpha=0.6
    )

    axes[0, 0].set_title("Original Data")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")
    axes[0, 0].legend()
    axes[0, 0].grid(True)


    # First transformation
    axes[0, 1].scatter(
        transformed_0[:, 0],
        transformed_0[:, 1],
        label="Class 0",
        marker="o",
        alpha=0.6
    )

    axes[0, 1].scatter(
        transformed_1[:, 0],
        transformed_1[:, 1],
        label="Class 1",
        marker="^",
        alpha=0.6
    )

    axes[0, 1].set_title("After First Transformation")
    axes[0, 1].set_xlabel("x1'")
    axes[0, 1].set_ylabel("x2'")
    axes[0, 1].legend()
    axes[0, 1].grid(True)


    # Whitening
    axes[1, 0].scatter(
        whitened_0[:, 0],
        whitened_0[:, 1],
        label="Class 0",
        marker="o",
        alpha=0.6
    )

    axes[1, 0].scatter(
        whitened_1[:, 0],
        whitened_1[:, 1],
        label="Class 1",
        marker="^",
        alpha=0.6
    )

    axes[1, 0].set_title("After Whitening")
    axes[1, 0].set_xlabel("x1''")
    axes[1, 0].set_ylabel("x2''")
    axes[1, 0].legend()
    axes[1, 0].grid(True)


    # Final
    axes[1, 1].scatter(
        final_0[:, 0],
        final_0[:, 1],
        label="Class 0",
        marker="o",
        alpha=0.6
    )

    axes[1, 1].scatter(
        final_1[:, 0],
        final_1[:, 1],
        label="Class 1",
        marker="^",
        alpha=0.6
    )

    axes[1, 1].set_title("After Final Transformation")
    axes[1, 1].set_xlabel("x1'''")
    axes[1, 1].set_ylabel("x2'''")
    axes[1, 1].legend()
    axes[1, 1].grid(True)


    # Add dataset name to the overall figure
    fig.suptitle(
        file_name.replace(".csv", ""),
        fontsize=16
    )

    plt.tight_layout()


    # --------------------------------------------------
    # Step 10: Save the plot
    # --------------------------------------------------

    plot_name = file_name.replace(
        ".csv",
        "_transformations.png"
    )

    plot_file = plot_folder / plot_name

    plt.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPlot saved to:")
    print(plot_file)


# --------------------------------------------------
# Run the procedure on all three datasets
# --------------------------------------------------

datasets = [
    "twogaussians33.csv",
    "twogaussians42.csv",
    "moons1.csv"
]

for dataset in datasets:
    simultaneous_diagonalization(dataset)

    #  question 2(f) is remaining