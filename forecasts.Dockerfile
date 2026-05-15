FROM python:3.13-slim-bookworm

WORKDIR /forecasts

COPY requirements.txt .

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# to prevent the immediate exit with return code of 0 
# and needless execution of the streamlit command
RUN echo "#!/bin/sh\n\
    python scripts/api/main.py &\n\
    streamlit run scripts/app/home/daily_forecasts.py --server.port=8501 --server.address=0.0.0.0 &\n\
    wait -n" > /forecasts/forecasting.sh

RUN chmod u+x /forecasts/forecasting.sh

EXPOSE 5050
EXPOSE 8501

ENTRYPOINT [ "/forecasts/forecasting.sh" ]