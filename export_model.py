import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--scaler", required=True)
parser.add_argument("--encoder", required=True)
parser.add_argument("--labels", required=True)

args = parser.parse_args()

os.makedirs("model", exist_ok=True)

shutil.copy2(args.model, "model/dnn_ddos_model.h5")
shutil.copy2(args.scaler, "model/scaler.pkl")
shutil.copy2(args.encoder, "model/label_encoder.pkl")

labels = args.labels.split(",")
with open("model/labels.txt", "w") as f:
    for l in labels:
        f.write(l.strip() + "\n")

print("✔ Model exported successfully")