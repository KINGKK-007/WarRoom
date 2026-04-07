FROM python:3.11-slim

# Set up a non-root user per HF Spaces requirements
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# CRITICAL: Remove the local openenv.py shim so the pip-installed
# openenv package is imported instead. The shim shadows the real
# package due to Python's module resolution order.
RUN rm -f openenv.py

EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "environment.server:app", "--host", "0.0.0.0", "--port", "7860"]
