FROM rust:1.75-bookworm as builder

WORKDIR /app
COPY . .

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y ca-certificates tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/momo /usr/local/bin/momo

ENV MOMO_HOST=0.0.0.0
ENV MOMO_PORT=3000
ENV DATABASE_URL=file:/data/momo.db

EXPOSE 3000

VOLUME ["/data"]

CMD ["momo"]
