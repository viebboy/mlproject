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

# Set the user to the created user
USER ${USERNAME}

# Ensure pip is up-to-date
RUN python -m pip install --upgrade pip

# swift loader
RUN git clone https://github.com/viebboy/swift-loader.git /swift-loader
RUN pip install --no-cache-dir -r /swift-loader/requirements.txt
RUN pip install --no-cache-dir -e /swift-loader

# install mlproject
COPY --chown=${USERNAME}:${GROUPNAME} ./ /mlproject
RUN pip install --no-cache-dir -r /mlproject/requirements.txt
RUN pip install --no-cache-dir -e /mlproject

# install model-training
RUN pip install --no-cache-dir -r /mlproject/cli/model-training/requirements.txt
RUN pip install --no-cache-dir -e /mlproject/cli/model-training

WORKDIR /${PACKAGE_NAME}
