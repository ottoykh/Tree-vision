import requests
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/OttoYu/Tree-Inspection"
headers = {"Authorization": "Bearer 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"}

def TreeAI(image_path):
    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query(image_path)

    if "error" in output:
        print("Error:", output["error"])
    else:
        for result in output:
            label = result["label"]
            confidence = result["score"]
            print("Prediction:", label, ",", confidence, "%")

    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def TreeAI_Batch(folder_path, output_csv):
    image_paths = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(folder_path, filename))

    num_images = len(image_paths)
    results = []

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{num_images}...")

        output = query(image_path)

        if "error" in output:
            print("Error:", output["error"])
        else:
            for result in output:
                filename = os.path.basename(image_path)
                label = result["label"]
                confidence = result["score"]
                results.append([filename, label, confidence])

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Prediction", "Confidence"])
        writer.writerows(results)