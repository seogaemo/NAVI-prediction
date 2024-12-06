ARG ARCH

FROM ultralytics/yolov5:latest${ARCH}

ENV TZ Asia/Seoul
ENV PYTHONIOENCODING UTF-8
ENV LC_CTYPE C.UTF-8

LABEL maintainer="https://suk.kr"

RUN python3 -m pip install flask requests

COPY . .

CMD [ "python3", "./main.py" ]