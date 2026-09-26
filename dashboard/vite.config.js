import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import seo from './vite-plugin-seo'

// Backend target for the dev proxy. Defaults to the docker-compose service
// name; set VITE_PROXY_TARGET=http://localhost:8000 to run the dev server on
// the host against a backend reachable at localhost (no CORS, same-origin).
const backend = process.env.VITE_PROXY_TARGET || 'http://backend:8000'
const renderer = process.env.VITE_RENDER_TARGET || 'http://renderer:3100'

// https://vitejs.dev/config/
export default defineConfig({
  // seo() runs on build only. It injects the crawler-visible homepage content
  // into #root and emits the static /alternatives pages, sitemap.xml and
  // llms.txt. See vite-plugin-seo.js.
  plugins: [react(), seo()],
  server: {
    allowedHosts: [
      'openshorts.app',
      'www.openshorts.app'
    ],
    // The dev container bind-mounts ./dashboard from the host, and inotify
    // events do NOT cross that boundary on Docker Desktop for Windows or
    // macOS. Without polling, Vite never learns a file changed: it keeps
    // serving the transform it cached at startup, HMR never fires, and a
    // browser refresh returns the same stale module. That failure is silent
    // and costs an edit-debug cycle every time, because the file on disk and
    // inside the container is plainly correct — only what Vite SERVES is old
    // (curl the module path to see it). Polling is the standard cost of
    // developing on a bind mount; VITE_POLL=0 turns it off on Linux, where
    // inotify works natively.
    watch: process.env.VITE_POLL === '0' ? undefined : {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/videos': { target: backend, changeOrigin: true },
      '/thumbnails': { target: backend, changeOrigin: true },
      '/gallery': { target: backend, changeOrigin: true },
      '/video': { target: backend, changeOrigin: true },
      '/render': { target: renderer, changeOrigin: true },
    }
  }
})
