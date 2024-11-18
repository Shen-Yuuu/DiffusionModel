import kagglehub

# Download latest version
path = kagglehub.dataset_download("dataclusterlabs/potholes-or-cracks-on-road-image-dataset")

print("Path to dataset files:", path)