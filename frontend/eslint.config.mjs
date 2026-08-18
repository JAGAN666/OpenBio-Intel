import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Generated from Pydantic by `npm run sync-types` / `sync-types:landscape`
    // / `sync-types:catalysts` — never hand-edited, so linting them only
    // produces noise about their own eslint-disable header.
    "types/trial.ts",
    "types/landscape.ts",
    "types/catalysts.ts",
  ]),
]);

export default eslintConfig;
