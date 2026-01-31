FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e ".[server]"

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "examples.production.api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
