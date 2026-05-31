import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import ViteYaml from '@modyfi/vite-plugin-yaml';
import { fileURLToPath, URL } from 'node:url';
/**
 * Vite configuration for the Iteris UI.
 *
 * - `@` resolves to `src` so imports never use brittle relative paths.
 * - `vite-plugin-yaml` lets content config (`src/content/*.yaml`) import as typed objects,
 *   keeping all copy/metrics/model data out of component code (YAML-driven, no hardcoding).
 * - The dev server proxies `/api` to the FastAPI backend whose URL is supplied via env,
 *   so no backend host is ever hardcoded in the bundle.
 */
export default defineConfig(function (_a) {
    var _b;
    var mode = _a.mode;
    var apiTarget = (_b = process.env.VITE_API_PROXY_TARGET) !== null && _b !== void 0 ? _b : 'http://localhost:8000';
    return {
        plugins: [react(), ViteYaml()],
        resolve: {
            alias: {
                '@': fileURLToPath(new URL('./src', import.meta.url)),
            },
        },
        server: {
            port: 5173,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: function (p) { return p.replace(/^\/api/, ''); },
                },
            },
        },
        test: {
            globals: true,
            environment: 'jsdom',
            setupFiles: ['./src/test/setup.ts'],
            css: false,
        },
    };
});
