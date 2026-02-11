FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ONNXRUNTIME_VERSION=1.23.2

RUN set -eux; \
    apt_get_install() { \
      apt-get update; \
      apt-get install -y --no-install-recommends ca-certificates curl libtesseract5 tesseract-ocr tesseract-ocr-eng; \
    }; \
    apt_get_install || apt_get_install; \
    rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) ORT_ARCH="x64" ;; \
      arm64) ORT_ARCH="aarch64" ;; \
      *) echo "Unsupported TARGETARCH: $TARGETARCH"; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz" -o /tmp/onnxruntime.tgz; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp; \
    cp /tmp/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/; \
    ln -sf /usr/local/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/local/lib/libonnxruntime.so.1; \
    ln -sf /usr/local/lib/libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so; \
    ldconfig; \
    rm -rf /tmp/onnxruntime.tgz /tmp/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}

ARG BINARY_PATH=target/release/momo
COPY ${BINARY_PATH} /usr/local/bin/momo
RUN chmod +x /usr/local/bin/momo

RUN set -e; \
    ldd_output="$(ldd /usr/local/bin/momo 2>&1 || true)"; \
    printf '%s\n' "$ldd_output"; \
    if printf '%s\n' "$ldd_output" | grep -q "not a dynamic executable"; then \
      if ! /usr/local/bin/momo --version >/dev/null 2>&1; then \
        echo "Binary at /usr/local/bin/momo is not runnable on this Linux image. Build for linux target/arch before docker build."; \
        exit 1; \
      fi; \
      echo "Static binary detected. Skipping shared-library check."; \
    elif printf '%s\n' "$ldd_output" | grep -q "not found"; then \
      echo "Missing shared libraries for /usr/local/bin/momo"; \
      exit 1; \
    fi

RUN ldconfig -p | grep libonnxruntime.so

ENV MOMO_HOST=0.0.0.0
ENV MOMO_PORT=3000
ENV DATABASE_URL=file:/data/momo.db
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.1

EXPOSE 3000

VOLUME ["/data"]

CMD ["momo"]
