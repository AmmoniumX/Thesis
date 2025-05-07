import os
import tkinter as tk
from tkinter import ttk
import cv2
import time
import json
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from pathlib import Path
from functools import partial
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from typing import Callable, Optional, TypedDict

# Global variables
GRID_SHAPE = (8, 8)

# Path to directories
IMGS_DIR = Path(".", "images")
OUTPUT_DIR = Path(".", "output")
LINE_WIDTH = 8


def get_distinct_colors(amount: int) -> list[tuple[int, int, int]]:
    """
    Generates a list of unique and distinct colors.
    Each color is represented as an RGB tuple.
    """
    colors = []
    for i in range(amount):
        # Evenly distribute hues in the HSV color space
        hue = i / amount
        saturation = 1.0  # Full saturation for vibrant colors
        value = 1.0  # Full brightness
        # Convert HSV to RGB and scale to 0-255
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb_scaled = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_scaled)

    return colors


def plot_image(left: np.ndarray, right: np.ndarray, frame: tk.Frame) -> None:
    """
    Plots and saves the images side by side inside a Tkinter window.
    """
    # Clear the frame
    for widget in frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Plot the original image
    ax1.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot the processed image
    ax2.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    ax2.set_title("Processed Image")
    ax2.axis("off")

    # Place the figure in a Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def color_distance(color1, color2):
    """Custom distance metric for colors that puts more emphasis on hue differences"""
    # Convert to HSV for better perceptual distance
    c1 = np.array(color1, dtype=np.float32) / 255.0
    c2 = np.array(color2, dtype=np.float32) / 255.0

    # Simple weighted RGB distance that emphasizes differences in proportions
    diff = np.abs(c1 - c2)
    return np.sum(diff) * 255  # Scale back to 0-255 range

def color_distance_hsv_weighted(color1, color2):
    """Custom distance metric for colors that puts more emphasis on hue differences"""
    # Convert RGB to HSV
    c1_rgb = np.array(color1, dtype=np.float32) / 255.0
    c2_rgb = np.array(color2, dtype=np.float32) / 255.0
    
    # OpenCV expects RGB in 0-1 range for conversion to HSV
    c1_hsv = cv2.cvtColor(np.array([[c1_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    c2_hsv = cv2.cvtColor(np.array([[c2_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    
    # Handle the circular nature of hue (0-180 in OpenCV's HSV)
    h1, s1, v1 = c1_hsv
    h2, s2, v2 = c2_hsv
    
    # Calculate hue difference accounting for circularity
    h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2)) / 180.0
    
    # Calculate differences for saturation and value
    s_diff = abs(s1 - s2) / 255.0
    v_diff = abs(v1 - v2) / 255.0
    
    # Apply weights to emphasize hue differences
    # These weights can be adjusted based on your specific needs
    h_weight = 1.0
    s_weight = 1.0
    v_weight = 1.0
    
    # Compute weighted distance
    distance = abs(h_weight * h_diff + s_weight * s_diff + v_weight * v_diff)
    
    # Scale to 0-255 range for consistency with the rest of the code
    return distance * 255

def kmeans_clustering(
    image: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies KMeans clustering to the image and returns the labels and cluster centers.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        n_clusters (int): The number of clusters to form.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the labels for each pixel and number of clusters.
    """
    pixels = image.reshape(-1, 3)  # Reshape to a 2D array of pixels

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=42
    )  # Adjust n_clusters as needed
    kmeans.fit(pixels)
    labels = kmeans.labels_  # Get the cluster labels for each pixel
    # Map the labels back to the original pixel values
    labels = labels.reshape(image.shape[:2])
    # centers = kmeans.cluster_centers_  # Get the cluster centers (colors)
    return labels, np.unique(labels)

def dbscan_clustering_custom(
    image: np.ndarray,
    eps: float,
    min_samples: int,
    *,
    quantization: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """DBSCAN clustering with custom distance metric"""
    print(
        f"Running DBSCAN clustering with parameters eps={eps}, min_samples={min_samples}, quantization={quantization}"
    )
    shape = image.shape
    pixels = image.reshape(-1, 3)

    # Apply quantization if specified
    if quantization is not None:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        quantized_pixels = np.floor_divide(pixels, quantization) * quantization

        # Get unique quantized colors and their indices
        unique_colors, inverse_indices = np.unique(
            quantized_pixels, axis=0, return_inverse=True
        )
        print(f"Quantized colors: {len(unique_colors)}")

        # Compute distance matrix
        n_colors = len(unique_colors)
        distance_matrix = np.zeros((n_colors, n_colors))
        for i in range(n_colors):
            for j in range(i + 1, n_colors):
                dist = color_distance(unique_colors[i], unique_colors[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Run DBSCAN with precomputed distances
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed", algorithm="brute"
        )
        unique_labels = dbscan.fit_predict(distance_matrix)

        # Map the unique labels back to all pixels using the inverse indices
        labels = unique_labels[inverse_indices]
    else:
        pixels_uint8 = np.clip(pixels, 0, 255).astype(np.uint8)
        pixels_reshaped = pixels_uint8.reshape(1, -1, 3)

        # Define custom metric function for sklearn
        def custom_metric(X, Y):
            return color_distance(X, Y)

        # Run DBSCAN with custom metric
        dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, metric=custom_metric, algorithm="brute"
        )
        labels = dbscan.fit_predict(pixels_reshaped)

    # Reshape labels to match the 2D shape of the image
    unique_labels = np.unique(labels, axis=0)
    labels = labels.reshape(shape[:2])
    return labels, unique_labels


class CoverageData(TypedDict):
    global_coverage: float
    size_distribution: dict[str, int]
    cell_coverage: dict[str, float]


@dataclass
class ClusterResult:
    original_image: np.ndarray
    scaled_image: np.ndarray
    labels: np.ndarray
    label_ids: np.ndarray


def calculate_coverage(mask: np.ndarray, grid_shape: tuple[int, int]):
    # Calculate the coverage percentage for each cluster
    global_coverage = np.sum(mask) / (mask.shape[0] * mask.shape[1]) * 100

    # Calculate the coverage percentage for each grid cell
    mask_height, mask_width = mask.shape
    grid_height, grid_width = grid_shape
    cell_height = mask_height // grid_height
    cell_width = mask_width // grid_width
    cell_coverage: dict[str, float] = {}

    # Update the cell coverage calculation
    for i in range(grid_height):
        for j in range(grid_width):
            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            cell_mask[
                i * cell_height : (i + 1) * cell_height,
                j * cell_width : (j + 1) * cell_width,
            ] = 1
            cell_coverage[f"({i}, {j})"] = (
                np.sum(mask[cell_mask == 1])
                / (cell_mask.shape[0] * cell_mask.shape[1])
                * 100
            )

    # Calculate the size distribution of clusters
    # Find contours in the mask
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the size of each cluster
    # areas = [cv2.contourArea(cnt) for cnt in contours]
    # Remove large contours
    # areas = [area for area in areas if area < 1000]
    
    # # Get bins
    # bin_counts, bin_edges = np.histogram(areas, bins=20)
    # Create a histogram of cluster sizes
    # Create bins splitting logarithmically
    # bin_counts, bin_edges = np.histogram(areas, bins=np.logspace(np.log10(1), np.log10(max(areas)), 20))

    # fig, ax = plt.subplots()
    # ax.hist(areas, bins=bin_edges, alpha=0.7, color="blue", edgecolor="black")
    # ax.set_title("Cluster Size Distribution")
    # ax.set_xlabel("Area (pixels)")
    # ax.set_ylabel("Frequency")
    # plt.tight_layout()
    # plt.savefig("cluster_size_distribution.png")
    # plt.close(fig)

    # # Convert the histogram to a dictionary
    # size_distribution = {
    #     f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}": int(bin_counts[i])
    #     for i in range(len(bin_counts))
    # }

    return {"global_coverage": global_coverage, "cell_coverage": cell_coverage}
    # return {"global_coverage": global_coverage, "cell_coverage": cell_coverage, "size_distribution": size_distribution}


def analyze_image(
    in_path: Path,
    out_dir: Path,
    *,
    scale_factor: float = 1.0,
    clusterer: Callable[[np.ndarray], tuple[np.ndarray, int]],
) -> ClusterResult:
    """
    Takes a path to an image and returns a dictionary with the estimated coverage percent of each cell in the grid.
    """

    # Load the input image
    image = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if image is None:
        print("Failed to load image:", in_path)
        return
    print("Loaded image:", in_path)

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Scale down the image
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    scaled_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Cluster the pixel colors using the provided clustering method
    print("Clustering image...")
    start_time = time.time()
    labels, label_ids = clusterer(scaled_image)
    n_labels = len(label_ids)
    end_time = time.time()
    print(
        f"Clustering took {end_time - start_time:.2f} seconds. Found {n_labels} clusters."
    )

    return ClusterResult(
        original_image=image,
        scaled_image=scaled_image,
        labels=labels,
        label_ids=label_ids,
    )


def get_label_visualization(
    image: np.ndarray, labels: np.ndarray, label_ids: np.ndarray
) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
    colors = get_distinct_colors(len(label_ids))
    cluster_colors = {label: colors[i] for i, label in enumerate(label_ids)}

    # Paint all pixels with their cluster color
    for label in label_ids:
        image[(labels == label)] = cluster_colors[label]

    return image, cluster_colors


def draw_grid_lines(image: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Draws grid lines on the image based on the specified grid shape.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        grid_shape (tuple[int, int]): The shape of the grid (rows, columns).

    Returns:
        np.ndarray: The image with grid lines drawn.
    """
    height, width = image.shape[:2]
    cell_height = height // grid_shape[0]
    cell_width = width // grid_shape[1]

    # Draw horizontal lines
    for i in range(1, grid_shape[0]):
        cv2.line(
            image,
            (0, i * cell_height),
            (width, i * cell_height),
            (255, 255, 255),
            LINE_WIDTH,
        )

    # Draw vertical lines
    for j in range(1, grid_shape[1]):
        cv2.line(
            image,
            (j * cell_width, 0),
            (j * cell_width, height),
            (255, 255, 255),
            LINE_WIDTH,
        )

    return image


def main():
    # Start the Tkinter window
    window = tk.Tk()
    window.title("Particle Detection")
    window.state("zoomed")

    # Create a PanedWindow to organize the layout horizontally
    paned_window = ttk.PanedWindow(window, orient=tk.HORIZONTAL)
    paned_window.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add the image frame to the left
    image_frame = tk.Frame(paned_window, width=800)
    paned_window.add(image_frame, weight=1)

    # Add the options frame to the right
    options_frame = tk.Frame(paned_window)
    paned_window.add(options_frame, weight=3)

    # Keep the button frame at the bottom
    button_frame = tk.Frame(window)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    # Set the initial sash position to give more space to image_frame
    window.update()  # Force an update to get window width
    paned_window.sashpos(
        0, int(window.winfo_width() * 0.7)
    )  # Set sash at 70% of window width

    select_layers_label = tk.Label(
        options_frame,
        text="Select particle layers:",
        font=("Arial", 14),
        bg="#f0f0f0",
        fg="black",
    )
    select_layers_label.pack(side=tk.TOP, pady=10)
    listbox = None  # Placeholder for the listbox

    # Where we will store all transient data
    data_store = {}
    eps_var = tk.DoubleVar(value=8.0)
    min_samples_var = tk.IntVar(value=5)
    quantization_var = tk.IntVar(value=8)
    prepared_clusterer = partial(
        dbscan_clustering_custom,
        eps=eps_var.get(),
        min_samples=min_samples_var.get(),
        quantization=quantization_var.get(),
    )
    # prepared_clusterer = partial(
    #     kmeans_clustering,
    #     n_clusters=3
    # )

    in_path = None

    def select_image():
        # Open a file dialog to select an image file
        nonlocal in_path
        in_path = tk.filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
        )
        if not in_path:
            print("No file selected.")
            window.quit()
            return
        # Convert the selected path to a Path object
        in_path = Path(in_path)

        # Restart the process
        process_image()

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def process_image():
        # Analyze the selected image
        nonlocal prepared_clusterer
        result = analyze_image(
            in_path,
            OUTPUT_DIR,
            scale_factor=1.0,
            clusterer=prepared_clusterer,
        )

        processed_image, label_to_color = get_label_visualization(
            result.scaled_image, result.labels, result.label_ids
        )

        data_store["in_path"] = in_path
        data_store["result"] = result
        data_store["image"] = result.original_image
        data_store["processed_image"] = processed_image
        data_store["label_to_color"] = label_to_color

        plot_image(
            result.original_image,
            processed_image,
            image_frame,
        )

        # goto select_layers
        select_layers()

    def select_layers():
        # Read from data store
        image = data_store["image"]
        processed_image = data_store["processed_image"]
        label_to_color = data_store["label_to_color"]

        # Separate the processed image into layers
        layers = {}
        for label, color in label_to_color.items():
            mask = np.all(processed_image == color, axis=-1)
            layer = np.zeros_like(processed_image)
            layer[mask] = processed_image[mask]
            layers[label] = layer

        # Ask user to select multiple layers
        nonlocal listbox
        if listbox is not None:
            listbox.destroy()
        listbox = tk.Listbox(
            options_frame,
            selectmode=tk.MULTIPLE,
            font=("Arial", 12),
            bg="#f0f0f0",
            fg="black",
            width=40,
            height=20,
        )
        for label in layers.keys():
            listbox.insert(tk.END, f"Layer {label}")
            listbox.select_set(tk.END)  # Select all layers by default

        def on_select_changed(event):
            selected_indices = listbox.curselection()

            # Update the processed image with the selected layers
            selected_labels = [int(listbox.get(i).split()[1]) for i in selected_indices]
            combined_image = np.zeros_like(processed_image)
            for label in selected_labels:
                if label in layers:
                    combined_image += layers[label]
            data_store["selected_labels"] = selected_labels

            plot_image(
                image,
                combined_image,
                image_frame,
            )

        listbox.bind("<<ListboxSelect>>", on_select_changed)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def get_parameters_dialog():
        # Create a new window for parameter input
        param_window = tk.Toplevel(window)
        param_window.title("Cluster Parameters")
        param_window.geometry("300x300")

        # Create input fields for parameters
        eps_label = tk.Label(param_window, text="Epsilon (eps):")
        eps_label.pack(pady=5)
        eps_entry = tk.Entry(param_window, textvariable=eps_var)
        eps_entry.pack(pady=5)

        min_samples_label = tk.Label(param_window, text="Min Samples:")
        min_samples_label.pack(pady=5)
        min_samples_entry = tk.Entry(param_window, textvariable=min_samples_var)
        min_samples_entry.pack(pady=5)

        quantization_label = tk.Label(param_window, text="Quantization:")
        quantization_label.pack(pady=5)
        quantization_entry = tk.Entry(param_window, textvariable=quantization_var)
        quantization_entry.pack(pady=5)

        def on_ok():
            # Update the clusterer with new parameters
            nonlocal prepared_clusterer
            prepared_clusterer = partial(
                dbscan_clustering_custom,
                eps=eps_var.get(),
                min_samples=min_samples_var.get(),
                quantization=quantization_var.get(),
            )
            param_window.destroy()
            # Go back to process_image
            process_image()

        ok_button = tk.Button(
            param_window,
            text="OK",
            command=on_ok,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
        )
        ok_button.pack(pady=10)

    select_image_button = tk.Button(
        button_frame,
        text="Select Image",
        command=select_image,
        font=("Arial", 12),
        bg="#2196F3",
        fg="white",
        padx=20,
        pady=10,
    )
    select_image_button.pack(side=tk.LEFT, padx=20)

    retry_button = tk.Button(
        button_frame,
        text="Retry",
        command=get_parameters_dialog,
        font=("Arial", 12),
        bg="#f44336",
        fg="white",
        padx=20,
        pady=10,
    )
    retry_button.pack(side=tk.LEFT, padx=20)

    def save():
        if "result" not in data_store:
            print("Not ready yet.")
            return

        out_path = tk.filedialog.asksaveasfilename(
            title="Save image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if not out_path:
            print("No file selected.")
            return
        out_path = Path(out_path)
        print("Saving image to:", out_path)

        print("Creating final image...")
        # Read from data store
        labels = data_store["result"].labels
        selected_labels = data_store["selected_labels"]
        binary_mask = np.zeros_like(labels, dtype=np.uint8)
        for label in selected_labels:
            binary_mask[labels == label] = 1

        final_image = np.zeros(
            shape=(labels.shape[0], labels.shape[1], 3), dtype=np.uint8
        )
        final_image[binary_mask == 1] = [0, 0, 255]  # Red
        # draw_grid_lines(final_image, GRID_SHAPE)
        cv2.imwrite(out_path, final_image)
        print("Final image saved to:", out_path)

        print("Computing coverage...")
        coverage_data = calculate_coverage(binary_mask, GRID_SHAPE)
        json_path = out_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(coverage_data, f, indent=4)

        print("Coverage data saved to:", json_path)

        print("Done.")
        # Close the window after saving
        window.quit()

    save_button = tk.Button(
        button_frame,
        text="Save",
        command=save,
        font=("Arial", 12),
        bg="#4CAF50",
        fg="white",
        padx=20,
        pady=10,
    )
    save_button.pack(side=tk.RIGHT, padx=20)

    # Start with processing the first image
    select_image()

    window.mainloop()


if __name__ == "__main__":
    main()
