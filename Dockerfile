FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY kenosis_chat.py /app/kenosis_chat.py

# Run as a non-root user so an app compromise doesn't get root in the container.
# NOTE: the mounted volumes must be writable by UID 10001 — on the host run once:
#   sudo chown -R 10001:10001 ./data ./backups
RUN useradd --system --uid 10001 --no-create-home oracle \
    && mkdir -p /data /backups \
    && chown oracle:oracle /app /data /backups
USER oracle

# The SQLite database lives on a mounted volume so it survives container rebuilds.
# Backups go to a separate volume so a copy of /data alone never includes them.
ENV KENOSIS_DB=/data/chat.db \
    KENOSIS_BACKUP_DIR=/backups \
    KENOSIS_PORT=8770
VOLUME ["/data", "/backups"]
EXPOSE 8770

HEALTHCHECK --interval=60s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/healthz',timeout=4).status==200 else 1)"

CMD ["python", "kenosis_chat.py"]
