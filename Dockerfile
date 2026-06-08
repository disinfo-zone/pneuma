FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir requests pypdf

COPY kenosis_chat.py /app/kenosis_chat.py

# The SQLite database lives on a mounted volume so it survives container rebuilds.
ENV KENOSIS_DB=/data/chat.db \
    KENOSIS_PORT=8770
VOLUME ["/data"]
EXPOSE 8770

CMD ["python", "kenosis_chat.py"]
