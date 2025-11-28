FROM python:3.10-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Let Google configure the port
ENV PORT=8080

# Run command (uses the $PORT variable automatically)
CMD exec gunicorn app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind :$PORT --timeout 0
