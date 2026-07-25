FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
# Checkpoint đã train phải được mount/copy vào models/ trước khi chạy container,
# vd: docker run -v $(pwd)/models:/app/models ...
COPY models/README.md models/README.md

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
