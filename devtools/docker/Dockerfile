FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1 \
    JUPYTER_ENABLE_LAB=yes

USER root
RUN apt-get update && apt-get install -y \
    git wget curl vim nano htop build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG NB_USER=developer
ARG NB_UID=1000
RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} && \
    chown -R ${NB_USER}:${NB_USER} /opt/conda
USER ${NB_USER}

WORKDIR /workspace
VOLUME ["/workspace"]

VOLUME ["/home/${NB_USER}/.jupyter", "/home/${NB_USER}/.local"]

COPY devtools/conda-envs/test_env.lock.yaml /tmp/docker_env.yaml

RUN conda update -n base -c defaults conda -y && conda clean -afy
RUN conda env create -n kinoml -f /tmp/docker_env.yaml && \
    conda clean -afy

COPY --chown=${NB_USER}:${NB_USER} . /workspace/

RUN conda run -n kinoml python -m pip install -e .

RUN conda run -n kinoml jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /home/${NB_USER}/.jupyter/jupyter_lab_config.py

RUN echo '#!/bin/bash\n\
set -e\n\
eval "$(conda shell.bash hook)"\n\
conda activate kinoml\n\
exec "$@"' > /home/${NB_USER}/entrypoint.sh && \
    chmod +x /home/${NB_USER}/entrypoint.sh

ENTRYPOINT ["/home/developer/entrypoint.sh"]

EXPOSE 8888

USER root
RUN mkdir -p /home/${NB_USER}/.jupyter /home/${NB_USER}/.local && \
    chown -R ${NB_USER}:${NB_USER} /home/${NB_USER}/.jupyter /home/${NB_USER}/.local
USER ${NB_USER}

CMD ["jupyter", "lab", "--no-browser"]
