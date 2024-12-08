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


TARGET_CONFIDENCE = 0.2


def getImage(id):
    # S3 이미지 URL 생성
    imageURL = f"{os.getenv('S3_URL')}/{id}.jpg"

    # 이미지 데이터 요청
    response = requests.get(imageURL)
    if response.status_code != 200:
        return None

    frameData = response.content
    return cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)


@app.route("/image", methods=["GET"])
def getPredictedImage():
    id = request.args.get("id")

    frameNp = getImage(id)
    framePil = Image.fromarray(frameNp)

    results = model(framePil)

    for bbox in zip(results.xyxy[0]):
        xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()

        if conf > TARGET_CONFIDENCE:
            cv2.rectangle(
                frameNp,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                (255, 0, 0),
                2,
            )

    _, buffer = cv2.imencode(".jpg", frameNp)
    response = buffer.tobytes()
    return response, 200, {"Content-Type": "image/jpeg"}


@app.route("/detect", methods=["GET"])
def detect():
    id = request.args.get("id")

    frameNp = getImage(id)
    framePil = Image.fromarray(frameNp)

    results = model(framePil)

    annos = []

    for bbox in zip(results.xyxy[0]):
        xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()

        if conf > TARGET_CONFIDENCE:
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
