/**
 * Standard Vite + React + TypeScript ESLint config (legacy .eslintrc format —
 * this project pins eslint@^8.57.1, whose default config format is .eslintrc,
 * not the flat eslint.config.js introduced as the default in eslint@9).
 *
 * Uses only the eslint-related devDependencies already in package.json
 * (@typescript-eslint/{eslint-plugin,parser}, eslint-plugin-react-hooks,
 * eslint-plugin-react-refresh) — `npm run lint` referenced this config
 * without it existing, so `lint` (and therefore CI) could not actually run.
 */
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'node_modules'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    // Matches this codebase's existing convention (e.g. test mocks) of
    // prefixing an intentionally-unused parameter with `_`.
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
  },
}
