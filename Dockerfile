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


# Copy the requirement file
COPY environment.yml .

# Create conda environment from yml file
RUN conda env create -f environment.yml

# Make RUN commands use the conda environment
SHELL ["conda", "run", "-n", "graph-aug", "/bin/bash", "-c"]

# Copy the source code
COPY . .

# Set up entry point to properly activate conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "graph-aug"]

# Run the application
CMD ["bash"]
