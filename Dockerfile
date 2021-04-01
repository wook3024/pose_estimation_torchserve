FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get install sudo && \
    apt-get install nano && \
    apt-get -y install git && \
    apt-get install wget && \
    apt-get -y install curl && \
    sudo apt install --no-install-recommends -y openjdk-11-jre-headless

RUN pip3 install torchserve torch-model-archiver && \
    pip3 install opencv-python

RUN git clone https://github.com/pytorch/serve.git /workspace/serve && \
    cd /workspace/serve && \
    python ./ts_scripts/install_dependencies.py --environment=dev && \
    python ./ts_scripts/install_from_src.py && \
    cd /workspace

RUN git clone https://github.com/wook3024/pose_estimation_torchserve.git /workspace/pose_estimation_torchserve

RUN cp -r /workspace/pose_estimation_torchserve/custom_handler /opt/conda/lib/python3.8/site-packages/ts/

RUN mkdir model_store && \
    curl -O https://download.pytorch.org/models/densenet161-8d451a50.pth

RUN python save_to_jit.py

RUN cd /workspace/pose_estimation_torchserve/build_model && \
    torch-model-archiver --model-name "PoseEstimation" --version 1.0 --serialized-file ./PoseEstimation_model.pt --handler "./handler.py"

RUN mv PoseEstimation.mar /workspace/model_store/PoseEstimation.mar && \
    cd /workspace

RUN torchserve --start --ncs --model-store model_store --models PoseEstimation_model.mar