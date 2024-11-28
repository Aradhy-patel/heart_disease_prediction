from src.heart_disease.preprocess import main
from src.heart_disease.train import train
import yaml
prep=yaml.safe_load(open("params.yaml"))["preprocess"]
main(prep["path"],prep["target"],prep["processed_path"],prep["file_type"])
tr=yaml.safe_load(open("params.yaml"))["train"]
train(tr["train_path"],tr["target"])