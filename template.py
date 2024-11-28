import os
from pathlib import Path
import logging


#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')

project_name= "heart_disease"

list_of_files=[
  ".github/workflows/.gitkeep",
  f"src/{project_name}/__init__.py",
  f"src/{project_name}/train.py",
  f"src/{project_name}/preprocess.py",
  f"src/{project_name}/evaluate.py",
  f"src/{project_name}/prediction.py",
  "dvc.yaml",
  "params.yaml",
  "requirements.txt",
  "setup.py",
  "app.py",
  "main.py"

]

for filepath in list_of_files:
  filepath=Path(filepath)
  filedir,filename=os.path.split(filepath)

  if filedir!="":
    os.makedirs(filedir,exist_ok=True)
    logging.info(f"Creating directory: {filedir} for the file : {filename}")

  if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
    with open(filepath,"w") as f:
      pass
      logging.info(f"Creating empty file: {filepath}")

  else:
    logging.info(f"Filepath already exists: {filepath}")






