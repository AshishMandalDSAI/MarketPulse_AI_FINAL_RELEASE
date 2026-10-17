# MarketPulse AI - Dockerfile (Streamlit dashboard service)
FROM python:3.13-slim

WORKDIR /app

# System dependencies (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Demo data/model pipeline runs at container startup (not build time) via
# the entrypoint script, so bind-mounted host volumes work correctly on
# both fresh installs (empty host dirs) and existing installs (user data
# already present). See docker-entrypoint.sh for details.

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
