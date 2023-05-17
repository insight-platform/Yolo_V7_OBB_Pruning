FROM nvcr.io/nvidia/pytorch:23.04-py3 as base

RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -yq curl unzip sudo python3-pip && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash -&& \
    apt-get install -y screen build-essential git ffmpeg libsm6 libxext6 &&  \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt requirements-docker.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements-docker.txt


#RUN jupyter lab clean && jupyter lab build

ARG UID=1000
ARG GID=1000

RUN addgroup --gid ${GID} jupyter && \
adduser --uid ${UID} --gid ${GID} --disabled-password --shell /bin/bash jupyter && \
echo "jupyter:123" | chpasswd
RUN adduser jupyter sudo
RUN echo 'jupyter ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ARG APP_PATH=/opt/app
RUN mkdir -p $APP_PATH
ENV APP_PATH=$APP_PATH
ENV PYTHONPATH=$APP_PATH:$APP_PATH/external
RUN chown -R jupyter ${APP_PATH}
RUN chgrp -R jupyter ${APP_PATH}
WORKDIR ${APP_PATH}

COPY --chown=jupyter:jupyter logging.conf ${APP_PATH}

FROM base AS yolov7obb

RUN apt-get update && apt-get install -y nodejs &&  \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY --chown=jupyter:jupyter entrypoint.sh ${APP_PATH}

USER ${UID}:${GID}

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

FROM base AS pycharm
USER ${UID}:${GID}
