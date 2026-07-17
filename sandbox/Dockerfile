FROM python:3.11-slim

# Install ca-certificates for HTTPS/SSL support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* || true

# Set proper DNS servers - attempt to fix network resolution
RUN mkdir -p /etc/docker && \
    echo 'nameserver 8.8.8.8' > /etc/resolv.conf && \
    echo 'nameserver 8.8.4.4' >> /etc/resolv.conf && \
    echo 'nameserver 1.1.1.1' >> /etc/resolv.conf

# Upgrade pip and install common ML packages
RUN python -m pip install --upgrade --no-cache-dir pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    scipy || echo "Warning: Some packages failed to install, will retry at runtime"

WORKDIR /workspace