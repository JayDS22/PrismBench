FROM ubuntu:24.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade

# Install system utilities.
RUN apt install -y --no-install-recommends \
    sudo \
    curl \
    systemctl \
    gnupg \
    git \
    vim

# Install Python and build deps (venv requires python3-venv on Debian/Ubuntu).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libgomp1 \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter.
#RUN pip3 install jupyterlab jupyterlab_vim

# Install uv for package management.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install project specific packages.
# This project uses pip-compile-generated requirements.txt (not pyproject.toml
# / uv.lock) as its canonical install spec. We invoke uv in pip mode against
# an explicit venv so packages land in the right path.
COPY requirements.txt /app/
WORKDIR /app
RUN uv venv --python 3.11 /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# Use --python flag so uv pip installs into the venv we just created
# regardless of any default uv pip resolution behaviour.
RUN uv pip install --python /app/.venv/bin/python -r /app/requirements.txt

# Copy project files.
COPY . /app

RUN mkdir /install

# Config.
ADD etc_sudoers /install/
COPY etc_sudoers /etc/sudoers
COPY bashrc /root/.bashrc

# Report package versions.
ADD version.sh /install/
RUN /install/version.sh 2>&1 | tee version.log

# Jupyter.
EXPOSE 8888
