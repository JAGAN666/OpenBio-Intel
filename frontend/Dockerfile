# Next.js 16 App Router frontend -- three explicit stages so the final image
# never carries the dependency-install cache or build toolchain, only what
# `npm start` (next start) actually needs at runtime.

# ---- Stage 1: deps -----------------------------------------------------------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- Stage 2: builder ---------------------------------------------------------
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* variables are INLINED into the client-side JS bundle at BUILD
# time -- `next build` reads process.env.NEXT_PUBLIC_API_URL once, here, and
# bakes the literal value into the shipped bundle. Setting it only as a
# runtime `environment:` on the Stage 3 container (as a naive compose config
# would) has NO effect on already-built client code; it must be supplied as a
# build ARG so it's present during this RUN step. docker-compose.yml passes
# it via `build.args`.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build

# ---- Stage 3: runner (minimal) -------------------------------------------------
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# `next start` needs the build output, the production node_modules (next
# itself + react + the runtime deps), and package.json for `npm start` to
# resolve the "start" script -- devDependencies (typescript, eslint, the
# Tailwind toolchain) are never copied in, which is where the size reduction
# over a single-stage build actually comes from.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["npm", "start"]
