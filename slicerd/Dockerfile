# archive-slicer: slicer.py + geotiff2pmtiles + rclone behind a small job
# controller (REST + MCP) for the OMV NUC. Build context = this directory.
#   docker build -t archive-slicer:latest .

# --- geotiff2pmtiles b300c9e + the half-pixel sampling fix ------------------
FROM golang:1.27-bookworm AS g2p
ARG G2P_COMMIT=b300c9e669cf55202c2cb93924cc880a52b37ca3
RUN apt-get update \
 && apt-get install -y --no-install-recommends libwebp-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q . \
 && git remote add origin https://github.com/pspoerri/geotiff2pmtiles.git \
 && git fetch -q --depth 1 origin "$G2P_COMMIT" \
 && git checkout -q FETCH_HEAD
COPY g2p/g2p_b300c9e_halfpx.patch /tmp/halfpx.patch
RUN git apply /tmp/halfpx.patch \
 && CGO_ENABLED=1 go build -trimpath \
      -ldflags "-X main.version=b300c9e-halfpx -X main.commit=${G2P_COMMIT}" \
      -o /out/geotiff2pmtiles ./cmd/geotiff2pmtiles/ \
 && /out/geotiff2pmtiles -version

# --- go-pmtiles CLI (verify_align.py shells out to `pmtiles tile`) ----------
FROM golang:1.27-bookworm AS pmtiles
RUN CGO_ENABLED=0 GOBIN=/out go install github.com/protomaps/go-pmtiles@v1.31.2 \
 && mv /out/go-pmtiles /out/pmtiles

# --- runtime: GDAL 3.12.1 (same as the Mac venv) ----------------------------
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.1
# The base image ships an Apache Arrow apt repo whose key no longer verifies; we need nothing from it.
RUN rm -f /etc/apt/sources.list.d/*arrow* \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3-venv libwebp7 rclone tini procps \
 && rm -rf /var/lib/apt/lists/*
# System numpy/GDAL bindings are shared into the venv; never pip-install numpy
# (gdal_array is built against the image's numpy 1.26).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
 && /opt/venv/bin/python -c "import numpy, PIL, pyproj, pmtiles, mcp; from osgeo import gdal_array; print('numpy', numpy.__version__)"
COPY --from=g2p /out/geotiff2pmtiles /usr/local/bin/geotiff2pmtiles
COPY --from=pmtiles /out/pmtiles /usr/local/bin/pmtiles
# (create the dir first: COPY --chmod would give an implicit parent the file's mode)
RUN install -d -m 755 /opt/slicerd
COPY --chmod=644 g2p/g2p_b300c9e_halfpx.patch /opt/slicerd/g2p_b300c9e_halfpx.patch
COPY --chmod=755 app/ /opt/slicerd/app/
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/state/home \
    RCLONE_CONFIG=/secrets/rclone.conf
EXPOSE 8765
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "/opt/slicerd/app/server.py"]
