# ==============================================================================
# Dockerfile: Tumor-Fraction-Aware AI for Early MCED from Liquid Biopsy
# Multi-stage reproducible bioinformatics & deep learning runtime
# ==============================================================================

FROM condaforge/miniforge3:latest AS base

LABEL maintainer="MCED Capstone Team" \
      description="Bioinformatics runtime for low-pass cfDNA fragmentomics & TF-aware AI" \
      version="1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/envs/mced-cfdna/bin:$PATH

WORKDIR /workspace

# Install system dependencies, compilers, and htslib prereqs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    libcurl4-gnutls-dev \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy environment specification
COPY environment.yml /tmp/environment.yml

# Create conda environment with bioinformatics & ML toolchain
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate conda environment in bash shell
SHELL ["conda", "run", "-n", "mced-cfdna", "/bin/bash", "-c"]

# Install ichorCNA and R dependencies (Bioconductor HMMcopy + ichorCNA)
# Note: ichorCNA is the gold-standard baseline for low-pass WGS CNA tumor fraction
RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev && \
    R -e "if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager', repos='https://cloud.r-project.org'); \
          BiocManager::install(c('HMMcopy', 'GenomeInfoDb', 'GenomicRanges'), update=FALSE, ask=FALSE);" && \
    R -e "install.packages('devtools', repos='https://cloud.r-project.org'); \
          devtools::install_github('GavinHaLab/ichorCNA');" && \
    rm -rf /var/lib/apt/lists/*

# Set default directory and permissions
COPY . /workspace

EXPOSE 8501

# Default command: display environment verification
CMD ["conda", "run", "--no-capture-output", "-n", "mced-cfdna", "python", "-c", "import torch, sklearn, xgboost; print('MCED environment initialized successfully.')"]
