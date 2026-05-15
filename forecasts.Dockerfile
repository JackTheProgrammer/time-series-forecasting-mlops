FROM python:3.13-slim-bookworm

WORKDIR /forecasts

COPY requirements.txt .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo "#!/bin/sh\n\
    (python scripts/api/main.py &) && \
    streamlit run scripts/app/home/home.py --server.port=8501 --server.address=0.0.0.0" > /forecasts/forecasting.sh

RUN chmod u+x /forecasts/forecasting.sh

EXPOSE 5050
EXPOSE 8501

ENTRYPOINT [ "/forecasts/forecasting.sh" ]