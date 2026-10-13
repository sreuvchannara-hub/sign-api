FROM python:3.11-slim-bookworm

# libraries that OpenCV (installed together with MediaPipe) needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# visitors of the public site cannot change the taught words
ENV ALLOW_TEACHING=0

EXPOSE 7860
CMD gunicorn -b 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 app:app
