FROM node:20-slim

WORKDIR /app

# Install common npm packages
RUN npm install -g \
    lodash \
    axios \
    date-fns \
    uuid

COPY package.json .
RUN npm install

COPY server.js .

# Run as non-root user for security
USER node

EXPOSE 3000

CMD ["node", "server.js"]
