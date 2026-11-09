# Assignment 4.0 MLOps — Simple Project Explanation

## The project in one sentence

I trained an animal image classifier on my local Kubernetes cluster, tracked and registered the model with MLflow, uploaded the model release to Azure Blob Storage, and used GitHub Actions to build and push the API image to Azure Container Registry. The final deployment to Azure Container Apps was attempted, but the app could not authenticate to the private registry, so I did not confirm a working public API.

## What each tool does

- **Kubernetes** runs and manages containers. I used the Kubernetes cluster in Docker Desktop on my computer. The dataset and shared files were stored on a Persistent Volume Claim (PVC).
- **Kubeflow Pipelines** runs the machine-learning steps in order. Each step runs in a container, and Kubeflow shows the steps and their results in the pipeline UI.
- **MLflow** records the training settings, evaluation metrics, and model files. Its model registry keeps named versions of models, such as `animal-cnn` version 1.
- **Azure Blob Storage** stores the approved model release files.
- **Azure Container Registry (ACR)** stores the Docker image for the API.
- **Azure Container Apps** is where the API image is intended to run and receive HTTP requests.
- **GitHub Actions** automates the release steps. It signs in to Azure using GitHub OIDC, which provides a short-lived token instead of a stored Azure client secret.

## What happened, step by step

### 1. Prepare the data

The local dataset contained 3,000 images: 1,000 each for cats, dogs, and pandas. The pipeline resized images to 64 by 64 pixels. It split the images into 2,040 training images, 510 validation images, and 450 test images. The dataset itself is not included in this hand-in ZIP.

### 2. Run training with Kubeflow on Kubernetes

I submitted the animal-classification pipeline to Kubeflow. Kubeflow ran three steps:

1. Preprocess the animal images.
2. Split the images into training, validation, and test groups.
3. Train the CNN model and save the model output.

Kubernetes provided the machines and containers that ran these steps. Azure Machine Learning training compute was not used for this local Kubeflow route.

### 3. Track the run and register the model with MLflow

The training code sent its settings, metrics, and model artifacts to the MLflow server running in the local Kubernetes cluster. MLflow recorded the run and registered the model as `animal-cnn` version 1.

The registered run was a baseline. Its test accuracy was about 60% and its test macro F1 score was about 0.57. These results show that the pipeline ran; they do not mean the classifier is ready for production use.

### 4. Publish the model files to Azure Blob Storage

I exported the registered model and its metadata, then uploaded these files to the `model-releases` container in the `animal-cnn/v1/` folder:

- `model.keras` — the trained model weights and structure.
- `labels.json` — the class names used by the API.
- `preprocessing.json` — the image size and color settings expected by the model.

Keeping the model release in Blob Storage gives the release workflow a known place from which to download it.

### 5. Package the API and publish its image

The FastAPI application loads the model and metadata, accepts an image at `/upload/image`, and has a `/health` endpoint. Docker packages the API and model files into one container image.

The GitHub Actions workflow downloads the three release files from Blob Storage, builds a Linux image, and pushes the image to `mlopsrafaacr.azurecr.io`. GitHub Actions uses OIDC to sign in to Azure without a saved client secret.

### 6. Attempt to run the API in Azure Container Apps

The Azure Container Apps environment and the `animal-api` app were created. The app uses the Express environment, which requires a user-assigned identity for pulling the private image. I created `animal-api-pull`, assigned it `AcrPull` on the registry, attached it to the app, and configured the registry entry to use it.

The latest deployment still failed while Azure tried to pull the image. The error asked for registry credentials or a managed identity client ID. Therefore, the container revision did not start, and I did not confirm that `/health` or `/upload/image` works on a public Azure endpoint.

## What was completed

- The local Kubeflow pipeline completed successfully.
- MLflow tracked the run and registered `animal-cnn` version 1.
- The model and its metadata were uploaded to Azure Blob Storage.
- GitHub Actions signed in to Azure, downloaded the release files, and built and pushed the API image to ACR.

## What remains incomplete

The Azure Container Apps revision could not authenticate to ACR and pull the private image. The final live API deployment and endpoint test were not completed. The workflow and Azure resources are included so this remaining deployment issue can be reviewed.

## Short explanation for a presentation

“Kubernetes ran the training containers on my computer. Kubeflow organized preprocessing, data splitting, and training as a repeatable pipeline. MLflow saved the run results and registered model version 1. I uploaded that model release to Azure Blob Storage. GitHub Actions used OIDC to download the release, build the API image, and push it to Azure Container Registry. I attempted to run it in Azure Container Apps, but its revision could not authenticate to the private registry, so the public API was not confirmed working.”
