# Dockerfile

FROM python:3.12-slim

WORKDIR /app

# git is needed both for the clones below and for the git+ dependency in
# requirements_docker.txt, so it stays in the image.
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Grid2op scenario data.
# Clone, copy and cleanup must stay in a SINGLE layer: a `git clone` in one RUN that is
# `rm -rf`'d in a later one still leaves the clone (including its .git history) in the
# intermediate layer, so it keeps counting towards the image size. --depth 1 avoids
# fetching the history in the first place.
RUN mkdir -p /root/data_grid2op && \
    git clone --depth 1 https://github.com/AI4REALNET/grid2op-scenario.git /tmp/grid2op-scenario && \
    cp -r /tmp/grid2op-scenario/ai4realnet_small /root/data_grid2op/ai4realnet_small && \
    rm -rf /tmp/grid2op-scenario

# ExpertOp4Grid -- same single-layer treatment.
RUN git clone --depth 1 https://github.com/Mleyliabadi/ExpertOp4Grid.git /tmp/ExpertOp4Grid && \
    pip install --no-cache-dir /tmp/ExpertOp4Grid/. && \
    rm -rf /tmp/ExpertOp4Grid

# Python dependencies -- CPU-only torch / TensorFlow, see requirements_docker.txt.
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy API + agent package
COPY app ./app/
COPY ExpertAgent ./ExpertAgent/
COPY setup.py .
COPY README.md .

# Install the project using setup.py at the root.
# EXPERTAGENT_CPU_ONLY makes setup.py require `tensorflow-cpu` rather than `tensorflow`,
# so this step cannot pull the GPU wheel back in on top of what was installed above.
# It is set per-RUN rather than via ENV so it does not linger in the final image.
RUN EXPERTAGENT_CPU_ONLY=1 pip install --no-cache-dir .

# Download the PPO model used by the API from the GitHub release into the image
# (see README: "Pretrained models and local_data").
ENV EXPERTAGENT_LOCAL_DATA=/app/local_data
RUN python -m ExpertAgent.utils.local_data download PPO_SB3

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
