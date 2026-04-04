FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# CRITICAL: Remove the local openenv.py shim so the pip-installed
# openenv package is imported instead. The shim shadows the real
# package due to Python's module resolution order.
RUN rm -f openenv.py

EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "7860"]
