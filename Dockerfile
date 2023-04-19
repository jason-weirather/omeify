#
#   Adapted from https://github.com/tebeka/pythonwise/blob/master/docker-miniconda/Dockerfile
#
#   miniconda vers: http://repo.continuum.io/miniconda
#   sample variations:
#     Miniconda3-latest-Linux-armv7l.sh
#     Miniconda3-latest-Linux-x86_64.sh
#     Miniconda3-py38_4.10.3-Linux-x86_64.sh
#     Miniconda3-py37_4.10.3-Linux-x86_64.sh
#
#   py vers: https://anaconda.org/anaconda/python/files
#   tf vers: https://anaconda.org/anaconda/tensorflow/files
#   tf-mkl vers: https://anaconda.org/anaconda/tensorflow-mkl/files
#

ARG UBUNTU_VER=22.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.10.10
ARG TF_VER=2.5.0

FROM ubuntu:${UBUNTU_VER}

# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim nano git

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
#ENV PATH=/miniconda/bin:${PATH}

# Add conda initialization to .bashrc
RUN echo "source /miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate base" >> ~/.bashrc

# Update conda and install packages within the base environment
RUN /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda update -y conda && \
    conda install -c anaconda -y python=${PY_VER}"

RUN mkdir /Source && \
    cd /Source && \
    git clone https://github.com/jason-weirather/bioformats2raw.git && \
    cd bioformats2raw && \
    git checkout 14dcd96

RUN cd /Source && \
    git clone https://github.com/jason-weirather/raw2ometiff.git && \
    cd raw2ometiff && \
    git checkout 57a2874

RUN /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda install -c free -y c-blosc=1.10.2 && \
    conda install -c conda-forge gradle=7.4.2 openjdk=8 numpy pandas"

RUN /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    cd /Source/bioformats2raw && \
    ./gradlew clean shadowJar createBioformats2rawScript"

RUN /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    cd /Source/raw2ometiff && \
    ./gradlew clean shadowJar createRaw2ometiffScript"

ENV PATH="/Source/bioformats2raw/build/bin:${PATH}"

ENV PATH="/Source/raw2ometiff/build/bin:${PATH}"

RUN mkdir /Source/omeify

COPY . /Source/omeify

RUN cd /Source/omeify && \
    /bin/bash -c "source /miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    pip install ."

#ENTRYPOINT ["bash", "-c", "source /miniconda/etc/profile.d/conda.sh && conda activate base && $0 \"$@\"", "--"]

RUN chmod +x /Source/omeify/entrypoint.sh
RUN chmod -R a+rx /Source/bioformats2raw/build/bin
RUN chmod -R a+rx /Source/raw2ometiff/build/bin


ENTRYPOINT ["/Source/omeify/entrypoint.sh"]
