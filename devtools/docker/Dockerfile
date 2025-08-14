# === Base args for flexibility ===
ARG MICROMAMBA_TAG=2.3-cuda11.7.1-ubuntu20.04
ARG USE_CUDA=false

# === Base image with micromamba ===
FROM mambaorg/micromamba:${MICROMAMBA_TAG}

# === Environment variables ===
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    PYTHONUNBUFFERED=1 \
    JUPYTER_ENABLE_LAB=yes

# === Install system dependencies as root ===
USER root
RUN apt-get update && apt-get install -y \
    git wget curl vim nano htop build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# === Create a non-root user for security ===
ARG NB_USER=developer
ARG NB_UID=1000
RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} && \
    chown -R ${NB_USER}:${NB_USER} /opt/conda
USER ${NB_USER}

# === Working directory (persistent) ===
WORKDIR /workspace
VOLUME ["/workspace"]

# Optional persistence for Jupyter settings/extensions
VOLUME ["/home/${NB_USER}/.jupyter", "/home/${NB_USER}/.local"]

# === Copy environment specs ===
COPY devtools/conda-envs/docker_env_cuda.yaml /tmp/docker_env_cuda.yaml
COPY devtools/conda-envs/docker_env_cpu.yaml /tmp/docker_env_cpu.yaml

# === Create environment ===
RUN micromamba update --all --yes && micromamba clean --all --yes
RUN if [ "$USE_CUDA" = "true" ]; then \
        micromamba create -n kinoml -f /tmp/docker_env_cuda.yaml -c conda-forge -c pytorch -c pyg -c openeye; \
    else \
        micromamba create -n kinoml -f /tmp/docker_env_cpu.yaml -c conda-forge -c pytorch -c pyg -c openeye; \
    fi && \
    micromamba clean --all --yes

# === Copy source (editable install for dev) ===
COPY --chown=${NB_USER}:${NB_USER} . /workspace/

RUN micromamba run -n kinoml python -m pip install -e .

# === Jupyter configuration ===
RUN micromamba run -n kinoml jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py

# === Entrypoint for env activation ===
RUN echo '#!/bin/bash\n\
set -e\n\
eval "$(micromamba shell hook --shell bash)"\n\
micromamba activate kinoml\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# === Expose Jupyter port ===
EXPOSE 8888

# === Default command: Jupyter Lab ===
CMD ["jupyter", "lab", "--no-browser"]
