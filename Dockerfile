# Use the base NVIDIA PyTorch container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Arguments for UID, GID, user name, and group name
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME
ARG PACKAGE_NAME


# Prevents interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  ca-certificates \
  libjpeg-dev \
  libpng-dev \
  libgl1 \
  libglib2.0-0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


# Create a group and user with the same UID, GID, user name, and group name as the host
RUN groupadd -g ${GID} ${GROUPNAME} && \
  useradd -m -u ${UID} -g ${GROUPNAME} ${USERNAME}

RUN mkdir /py_libs
RUN chown ${USERNAME}:${GROUPNAME} /py_libs

# Set the user to the created user
USER ${USERNAME}

# Ensure pip is up-to-date
RUN python -m pip install --upgrade pip

# swift loader
RUN git clone https://github.com/viebboy/swift-loader.git /py_libs/swift-loader
RUN pip install --no-cache-dir -r /py_libs/swift-loader/requirements.txt
RUN pip install --no-cache-dir -e /py_libs/swift-loader

# install mlproject
COPY --chown=${USERNAME}:${GROUPNAME} ./ /py_libs/mlproject
RUN pip install --no-cache-dir -r /py_libs/mlproject/requirements.txt
RUN pip install --no-cache-dir -e /py_libs/mlproject

# install model-training
RUN pip install --no-cache-dir -r /py_libs/mlproject/cli/model-training/requirements.txt
RUN pip install --no-cache-dir -e /py_libs/mlproject/cli/model-training

WORKDIR /
