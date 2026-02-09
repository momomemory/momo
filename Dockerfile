FROM rust:1-bookworm AS builder

WORKDIR /app
COPY . .

ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      pkg-config \
      libssl-dev \
      clang \
      cmake \
      libleptonica-dev \
      libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

RUN if [ "$TARGETARCH" = "arm64" ]; then \
      export WHISPER_GGML_NATIVE=OFF; \
      export WHISPER_GGML_CPU_ARM_ARCH=armv8.5-a; \
      export CMAKE_C_FLAGS="-march=armv8.5-a"; \
      export CMAKE_CXX_FLAGS="-march=armv8.5-a"; \
    fi; \
    cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/momo /usr/local/bin/momo

ENV MOMO_HOST=0.0.0.0
ENV MOMO_PORT=3000
ENV DATABASE_URL=file:/data/momo.db

EXPOSE 3000

VOLUME ["/data"]

CMD ["momo"]
