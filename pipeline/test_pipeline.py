from kfp import dsl

@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311")
# @dsl.component(base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301")
def say_hello(name: str, wow: str) -> str:
    hello_text = f'Hello, {name}!, wow: {wow}'
    print(hello_text)
    return hello_text

@dsl.pipeline
def hello_pipeline(recipient: str = 'yosi', wow: str = '') -> str:
    hello_task = say_hello(name=recipient, wow=wow)
    return hello_task.output

from kfp import compiler

compiler.Compiler().compile(hello_pipeline, 'pipeline4.yaml')