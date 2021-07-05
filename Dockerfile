FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
RUN apt-get update && apt-get install -y openssh-server nfs-common && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    (echo '123456'; echo '123456') | passwd root

RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ transforms3d \
        typing \
        numpy \
        keras \
        h5py \
        pandas \
        zipfile36 \
        scikit-optimize \
        pytest \
        sphinx \
        matplotlib \
        myst_parser \
        sphinx_rtd_theme==0.5.2 \
        tensorboard==2.4.1 \
        sympy==1.5.1 \
        pyevtk==1.1.1 \
        flask==1.1.2 \
        requests==2.25.0 \
        networkx==2.5.1
COPY . /idrlnet/
RUN cd /idrlnet && pip install -e .
ENTRYPOINT service ssh start && bash