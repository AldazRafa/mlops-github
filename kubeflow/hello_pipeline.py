from kfp import compiler, dsl

@dsl.component
def say_hello(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message

@dsl.pipeline(name="hello-pipeline")
def hello_pipeline(recipient: str = "Rafael"):
    say_hello(name=recipient)

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path="kubeflow/hello_pipeline.yaml",
    )