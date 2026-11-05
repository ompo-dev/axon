FROM rust:1.89-bookworm AS core

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY evals ./evals
COPY docs ./docs
COPY examples ./examples
COPY scripts ./scripts
RUN cargo build --release --features curriculum --bins

FROM oven/bun:1.4.2 AS ui

WORKDIR /app
COPY ui/package.json ui/bun.lock ./ui/
RUN cd ui && bun install --frozen-lockfile
COPY ui ./ui
RUN cd ui && bun run build

FROM oven/bun:1.4.2

WORKDIR /app
ENV NODE_ENV=production
ENV AXON_HOST=0.0.0.0
COPY --from=core /build/target/release ./target/release
COPY --from=core /build/src ./src
COPY --from=core /build/docs ./docs
COPY --from=core /build/evals ./evals
COPY --from=core /build/examples ./examples
COPY --from=ui /app/ui/package.json /app/ui/bun.lock ./ui/
RUN cd ui && bun install --frozen-lockfile --production
COPY --from=ui /app/ui/dist ./ui/dist
COPY --from=ui /app/ui/*.mjs ./ui/
COPY --from=ui /app/ui/*.js ./ui/

EXPOSE 4317
CMD ["bun", "ui/server.mjs", "4317"]
