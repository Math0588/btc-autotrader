FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY autotrader.py .

# State file will be stored in /data for persistence
RUN mkdir -p /data
ENV STATE_FILE=/data/autotrader_state.json

CMD ["python", "-u", "autotrader.py"]
