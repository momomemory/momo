# syntax=docker/dockerfile:1.7

FROM rust:1-bookworm AS chef

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      pkg-config \
      libssl-dev \
      clang \
      cmake \
      libleptonica-dev \
      libtesseract-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cargo install cargo-chef --locked

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json

COPY . .

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
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
