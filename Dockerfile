# Use Miniconda as base image
FROM continuumio/miniconda3:4.12.0

# Prevents Python from writing pyc files and keeps Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install system dependencies required for PyTorch Geometric
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.yml .

# Create conda environment from yml file
RUN conda env create -f requirements.yml

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "graph-aug", "/bin/bash", "-c"]

# Install pre-built PyTorch Geometric extensions to avoid compilation issues
RUN conda run -n graph-aug pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://pytorch-geometric.com/whl/torch-1.7.1+cu102.html

# Copy the source code
COPY . .

# Set up entry point to properly activate conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "graph-aug"]

# Run the application
CMD ["bash"]
