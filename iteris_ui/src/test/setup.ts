import '@testing-library/jest-dom/vitest';

// jsdom does not implement btoa; the mock API uses it to encode SVG mask layers.
if (typeof globalThis.btoa === 'undefined') {
  globalThis.btoa = (s: string) => Buffer.from(s, 'binary').toString('base64');
}

