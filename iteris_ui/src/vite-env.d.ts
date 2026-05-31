/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_PROXY_TARGET?: string;
  readonly VITE_USE_MOCKS?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** YAML content modules imported via vite-plugin-yaml resolve to plain objects. */
declare module '*.yaml' {
  const data: unknown;
  export default data;
}
