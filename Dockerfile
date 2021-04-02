FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && \
    apt-get install sudo && \
    apt-get install nano && \
    apt-get -y install git && \
    apt-get install wget && \
    apt-get -y install curl && \
    sudo apt install --no-install-recommends -y openjdk-11-jre-headless

RUN git clone https://github.com/pytorch/serve.git /workspace/serve && \
    cd /workspace/serve && \
    python ./ts_scripts/install_dependencies.py && \
    # python ./ts_scripts/install_dependencies.py --environment=dev && \
    # python ./ts_scripts/install_from_src.py && \
    cd /workspace

RUN pip3 install torchserve torch-model-archiver && \
    pip3 install opencv-python && \ 
    pip3 install -U Flask && \
    pip3 install shinuk


RUN git clone https://github.com/wook3024/pose_estimation_torchserve.git /workspace/pose_estimation_torchserve

# RUN cp -r /workspace/pose_estimation_torchserve/shinuk /opt/conda/lib/python3.8/site-packages/ts/

RUN mkdir model_store && \
    cd /workspace/pose_estimation_torchserve/build_model && \
    curl -O https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth

RUN cd /workspace/pose_estimation_torchserve/build_model && \
    python ./save_to_jit.py && \
    torch-model-archiver --model-name "PoseEstimation" --version 1.0 \
    --serialized-file ./PoseEstimation_model.pt \
    --handler "./handler.py" && \
    mv ./PoseEstimation.mar /workspace/model_store/PoseEstimation.mar && \
    cd /workspace

CMD ["torchserve", "--start", "--ncs", "--model-store", "model_store", "--models", "PoseEstimation.mar"]

# docker run --rm --name pytorch --gpus all -it wook3024/pose_estimation_torchserve:1.0.0
# torchserve --start --ncs --model-store model_store --ts-config pose_estimation_torchserve/config.properties --models PoseEstimation.mar
# curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
# curl http://127.0.0.1:8080/predictions/PoseEstimation -T kitten_small.jpg