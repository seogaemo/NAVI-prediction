import os
import cv2
import torch
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# YOLO 모델과 가중치 로드
model = torch.hub.load(
    "./", "custom", path="./models/best.pt", source="local", force_reload=True
)


@app.route("/detect", methods=["GET"])
def detect():
    id = request.args.get("id")

    # S3 이미지 URL 생성
    imageURL = f"{os.getenv('S3_URL')}/{id}.jpg"

    # 이미지 데이터 요청
    response = requests.get(imageURL)
    if response.status_code != 200:
        return jsonify({"error": "Image not found"}), 404

    frameData = response.content
    frameNp = cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)
    framePil = Image.fromarray(frameNp)

    results = model(framePil)

    annos = []

    for bbox in zip(results.xyxy[0]):
        xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()

        if conf > 0.2:
            annos.append(
                {
                    "xmin": int(xmin),
                    "ymin": int(ymin),
                    "xmax": int(xmax),
                    "ymax": int(ymax),
                    "label": int(label),
                    "confidence": float(conf),
                }
            )

    print(annos)

    return jsonify({"predictions": annos})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="9000")
