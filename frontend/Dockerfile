# 1) Build your app
FROM node:18-alpine AS build
WORKDIR /app

# only copy package files, install deps
COPY package.json package-lock.json ./
RUN npm ci

# copy everything (including public/) and build
COPY . .
RUN npm run build

# 2) Serve with nginx
FROM nginx:stable-alpine

# copy the entire dist folder
COPY --from=build /app/dist      /usr/share/nginx/html

# also pull in your fallback image from public/
COPY --from=build /app/public/fallback-image.png \
                     /usr/share/nginx/html/fallback-image.png

# your custom nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
