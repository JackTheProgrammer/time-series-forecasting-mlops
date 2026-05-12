FROM python:3.13-slim-bookworm

WORKDIR /forecasts

COPY requirements.txt .

RUN pip install --no-cache-dir extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo ""