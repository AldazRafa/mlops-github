"""Kubeflow pipeline for the local cats, dogs, and panda dataset."""

from kfp import compiler, dsl, kubernetes
from kfp.dsl import Model, Output


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["Pillow==12.3.0"],
)
def preprocess_animal_data() -> str:
    """Resize each class to the 64x64 input size expected by the CNN."""
    import os
    import subprocess
    import sys

    script = "/data/components/dataprep/code/dataprep.py"
    for label in ("cats", "dogs", "panda"):
        output_dir = f"/data/work/processed/{label}"
        os.makedirs(output_dir, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                script,
                "--data",
                f"/data/animals/{label}",
                "--output_data",
                output_dir,
            ],
            check=True,
        )
    return "Resized all class images to 64x64 pixels."


@dsl.component(base_image="python:3.11")
def split_animal_data() -> str:
    """Create a reproducible 80/20 split from the resized images."""
    import subprocess
    import sys

    script = "/data/components/dataprep/code/traintestsplit.py"
    subprocess.run(
        [
            sys.executable,
            script,
            "--datasets",
            "/data/work/processed/cats",
            "/data/work/processed/dogs",
            "/data/work/processed/panda",
            "--training_data_output",
            "/data/work/train",
            "--testing_data_output",
            "/data/work/test",
            "--validation_data_output",
            "/data/work/validation",
            "--split_size",
            "15",
            "--validation_size",
            "20",
        ],
        check=True,
    )
    return "Created an 80/20 split for cats, dogs, and pandas."


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "numpy==2.3.3",
        "scikit-learn==1.7.2",
        "scipy==1.16.2",
        "tensorflow==2.21.0",
        "Pillow==12.3.0",
        "mlflow==3.15.0",
    ],
)
def train_animal_model(epochs: int, model: Output[Model]):
    """Run the existing CNN training script and capture its model artifact."""
    import subprocess
    import sys
    import os

    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.kubeflow.svc.cluster.local:5000"
    os.environ["DATA_VERSION"] = "animals-local-v1"

    script = "/data/components/training/code/train.py"
    subprocess.run(
        [
            sys.executable,
            script,
            "--training_folder",
            "/data/work/train",
            "--testing_folder",
            "/data/work/test",
            "--validation_folder",
            "/data/work/validation",
            "--output_folder",
            model.path,
            "--epochs",
            str(epochs),
            "--model_name",
            "animal-cnn",
        ],
        check=True,
    )


@dsl.pipeline(
    name="animal-classification-local",
    description="Split the local animal images and train the existing CNN on Kubernetes.",
)
def animal_classification_pipeline(epochs: int = 5):
    preprocess_task = preprocess_animal_data().set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        preprocess_task,
        pvc_name="animals-data",
        mount_path="/data",
    )

    split_task = split_animal_data().after(preprocess_task)
    split_task.set_caching_options(enable_caching=False)
    kubernetes.mount_pvc(
        split_task,
        pvc_name="animals-data",
        mount_path="/data",
    )

    train_task = train_animal_model(epochs=epochs)
    train_task.after(split_task)
    train_task.set_caching_options(enable_caching=False)
    train_task.set_cpu_request("2")
    train_task.set_cpu_limit("4")
    train_task.set_memory_request("4Gi")
    train_task.set_memory_limit("8Gi")
    kubernetes.mount_pvc(
        train_task,
        pvc_name="animals-data",
        mount_path="/data",
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=animal_classification_pipeline,
        package_path="kubeflow/animal_pipeline.yaml",
    )
