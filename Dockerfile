FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libtesseract4 tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

ARG BINARY_PATH=target/release/momo
COPY ${BINARY_PATH} /usr/local/bin/momo
RUN chmod +x /usr/local/bin/momo

RUN set -e; \
    ldd /usr/local/bin/momo; \
    if ldd /usr/local/bin/momo | grep -q "not found"; then \
      echo "Missing shared libraries for /usr/local/bin/momo"; \
      exit 1; \
    fi

ENV MOMO_HOST=0.0.0.0
ENV MOMO_PORT=3000
ENV DATABASE_URL=file:/data/momo.db

EXPOSE 3000

VOLUME ["/data"]

CMD ["momo"]
