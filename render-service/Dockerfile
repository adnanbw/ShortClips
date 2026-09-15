FROM node:18-bookworm-slim

# Install Chromium and FFmpeg for Remotion rendering.
# The Noto packages are not decoration: Chromium renders a character it has no
# font for as a tofu box, so without them every Hindi, Arabic, Thai, Japanese,
# Korean or Chinese caption comes out as empty rectangles while the ASR text
# behind it is perfectly correct. fonts-liberation keeps the Latin metrics the
# existing English styles were designed against.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        chromium \
        ffmpeg \
        fontconfig \
        fonts-liberation \
        fonts-noto-core \
        fonts-noto-cjk \
        fonts-noto-color-emoji && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV REMOTION_BUNDLE_PATH=/app/remotion
ENV OUTPUT_DIR=/output

WORKDIR /app

# Copy render service source
COPY render-service/package.json ./
RUN npm install

COPY render-service/tsconfig.json ./
COPY render-service/src/ ./src/

# Copy remotion source for bundling
COPY remotion/package.json /app/remotion/package.json
COPY remotion/tsconfig.json /app/remotion/tsconfig.json
COPY remotion/src/ /app/remotion/src/
COPY remotion/public/ /app/remotion/public/

# Install remotion deps
RUN cd /app/remotion && npm install

# Build TypeScript
RUN npm run build

EXPOSE 3100
CMD ["node", "dist/server.js"]
