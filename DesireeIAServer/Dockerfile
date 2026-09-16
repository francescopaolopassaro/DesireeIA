# DesireeIA server container image.
#
# Build context MUST be the repository root (one level above this file),
# because this needs Build/Python/desireeia - the packaged wrapper that
# bundles the compiled linux-x64 native engine (see ../Build/build.ps1) -
# which lives outside DesireeIAServer/ itself:
#
#   docker build -f DesireeIAServer/Dockerfile -t desireeia-server .
#
# Build/build.ps1 must already have produced Build/Python before this
# runs; this image only packages what that script already built, it does
# not compile the native engine itself (no CUDA toolkit, no MSVC/MinGW,
# no CMake in this image - by design, it is a thin runtime layer).
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 desireeia

WORKDIR /app
COPY Build/Python /tmp/desireeia-python
COPY DesireeIAServer /tmp/desireeia-server

RUN pip install --no-cache-dir /tmp/desireeia-python /tmp/desireeia-server \
    && rm -rf /tmp/desireeia-python /tmp/desireeia-server

ENV DESIREEIA_DATA_DIR=/data
RUN mkdir -p /data && chown -R desireeia:desireeia /data
VOLUME ["/data"]

USER desireeia
EXPOSE 8080

ENTRYPOINT ["desireeia-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
