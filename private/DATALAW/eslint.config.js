export default [
  {
    ignores: ['dist/**', 'node_modules/**', 'src/i18n/locales/**', 'docs/**'],
  },
  {
    files: [
      'src/**/*.js',
      'src/**/*.ts',
      'src/**/*.tsx',
      'tests/**/*.ts',
      'tools/**/*.js',
      'vite.config.ts',
    ],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        window: 'readonly',
        document: 'readonly',
        console: 'readonly',
        AbortController: 'readonly',
        AbortSignal: 'readonly',
        DOMException: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        localStorage: 'readonly',
        crypto: 'readonly',
        process: 'readonly',
      },
    },
    rules: {},
  },
];
