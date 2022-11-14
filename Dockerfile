FROM python:3.10.6-slim

# Copy relevant files
COPY . /quadmompy
WORKDIR /quadmompy

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

ENTRYPOINT [ "bash" ]
