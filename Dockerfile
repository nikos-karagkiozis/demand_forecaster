# Dockerfile

# --- Stage 1: Builder ---
# This stage installs dependencies into a virtual environment using Poetry.
FROM python:3.12-slim AS builder

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Configure Poetry to create the virtual environment in the project's root
RUN poetry config virtualenvs.in-project true

# Copy the dependency definition files
COPY pyproject.toml poetry.lock* ./

# Install dependencies.
# --no-root: Don't install the project itself, only its dependencies.
# --only main: Exclude development dependencies and install only the main group.
RUN poetry install --no-root --only main


# --- Stage 2: Final ---
# This stage creates the final, lightweight image.
FROM python:3.12-slim AS final

# Set the working directory
WORKDIR /app

# System dependencies needed at runtime (e.g., LightGBM needs libgomp)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Activate the virtual environment by adding its bin to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY deploy/ ./deploy/

# Set the PYTHONPATH to ensure the 'sales_forecast' module is importable
ENV PYTHONPATH="/app/src:/app"