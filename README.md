# huggingface_sagemaker_pipeline

Your custom Huggingface model for MLOps pipeline in sagemaker.

# Overview
![overview](./img/overview.png)

# Pipeline
![pipeline](./img/pipeline.png)

# Prepare

1. Preprocess dockerfile
2. Inference dockerfile(cpu or cuda) [sagemaker-pytorch-inference-toolkit](https://github.com/aws/sagemaker-pytorch-inference-toolkit)
3. Python Script (train.py, evaluation.py, inference.py)

# Sagemaker Notebook
[Sagemaker_pipeline](https://github.com/philschmid/huggingface-sagemaker-workshop-series/blob/main/workshop_3_mlops/lab_1_sagemaker_pipeline.ipynb)

# Sagemaker Environment variables
[Environment variables](https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md)

# Custom script processing step
```
from sagemaker.processing import ScriptProcessor

processing_output_destination = 'YOUR S3 PATH'

custom_processor = ScriptProcessor(
    image_uri='YOUR ECR URI',
    instance_type='ml.c5.2xlarge',
    instance_count=1,
    base_job_name=base_job_prefix + "/preprocessing",
    sagemaker_session=sagemaker_session,
    role=role,
    command = ["python3"])

step_process = ProcessingStep(
    name="ProcessDataForTraining",
    cache_config=cache_config,
    processor=custom_processor,
    job_arguments=["--bucket",'hg.sage',
                   "--file_name", 'train/train.csv'],
    inputs=[
        ProcessingInput(
            input_name="train.csv",
            source=f"{processing_output_destination}/train",
            destination="/opt/ml/processing/input/train"
        ),
        ProcessingInput(
            input_name="test.csv",
            source=f"{processing_output_destination}/test",
            destination="/opt/ml/processing/input/test"
        ),
    ],

    outputs=[
        ProcessingOutput(
            output_name="train",
            destination=f"{processing_output_destination}/train",
            source="/opt/ml/processing/train",
        ),
        ProcessingOutput(
            output_name="test",
            destination=f"{processing_output_destination}/test",
            source="/opt/ml/processing/test",
        ),
    ],
    code=f'{YOUR SOURCE PATH}/Preprocess.py',
)
```
