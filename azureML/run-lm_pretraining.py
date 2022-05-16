# This is the AzureML run-script for lm pre-training

from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset

if __name__ == "__main__":
    ws = Workspace.from_config()
    datastore = ws.get_default_datastore()

    dataset_name = 'mlm_GIGA_1990_100_MBERT_tokenized_labeled_grouped_512_data_online'
    dataset = Dataset.get_by_name(workspace=ws, name=dataset_name)

    experiment = Experiment(workspace=ws, name='day6-lm-pt-mBERT-adapt1990-data_batch4_100') 
    config_directory = Dataset.File.from_files(path=(datastore, 'datasets/teacher_model_configs')) 

    config = ScriptRunConfig(
        source_directory='./src',
        script='lm_pretraining.py',
        compute_target='exjobbgpu',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            '--config_path', config_directory.as_named_input('input2').as_mount()
        ],
    )
    
    env = Environment.from_conda_specification(
        name='mbert_swe_azure',
        file_path='mbert_swe_azure.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)