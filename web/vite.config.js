import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build to jarvs/web_dist so the Python server (jarvs/server.py) can serve the
// production bundle. In dev, `npm run dev` runs Vite on 5173 and proxies the API
// to the Python backend on 8765, so the two run side by side.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../jarvs/web_dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
      },
    },
  },
});
