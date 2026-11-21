FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY demo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY harness /app/harness
COPY solution /app/solution
COPY demo/app.py /app/demo/app.py

ENV APP_ROOT=/app PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
CMD ["python", "demo/app.py"]
