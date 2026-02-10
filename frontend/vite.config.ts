import { defineConfig, loadEnv } from 'vite';
import preact from '@preact/preset-vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const target = env.VITE_DEV_API_ORIGIN || 'http://127.0.0.1:3000';

  return {
    plugins: [preact()],
    server: {
      proxy: {
        '/api': {
          target,
          changeOrigin: false,
        },
      },
    },
  };
});
