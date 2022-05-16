# This AzureML script will upload the teacher configuration and state files to a datastore

from azureml.core import Workspace
from azureml.core import Dataset
from azureml.data.datapath import DataPath

ws = Workspace.from_config()
datastore = ws.get_default_datastore()
Dataset.File.upload_directory(src_dir='src/teacher_model_configs', 
                              target=DataPath(datastore, "datasets/teacher_model_configs")
                             )
