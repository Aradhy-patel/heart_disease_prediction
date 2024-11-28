from src.heart_disease.preprocess import main
import yaml
prep=yaml.safe_load(open("params.yaml"))["preprocess"]
main(prep["path"],prep["target"],prep["processed_path"],prep["file_type"])