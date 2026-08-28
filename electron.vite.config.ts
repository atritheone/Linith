import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "electron-vite";

const packageMetadata = JSON.parse(readFileSync(resolve(__dirname, "package.json"), "utf8")) as {
  version: string;
};

export default defineConfig({
  main: {
    build: {
      outDir: "dist/main",
      rollupOptions: {
        input: {
          main: resolve(__dirname, "src/main/main.ts")
        }
      }
    }
  },
  preload: {
    build: {
      outDir: "dist/preload",
      rollupOptions: {
        input: {
          preload: resolve(__dirname, "src/preload/preload.ts")
        }
      }
    }
  },
  renderer: {
    root: resolve(__dirname, "src/renderer"),
    publicDir: resolve(__dirname, "build"),
    define: {
      __APP_VERSION__: JSON.stringify(packageMetadata.version)
    },
    build: {
      outDir: resolve(__dirname, "dist/renderer"),
      emptyOutDir: true
    }
  }
});
