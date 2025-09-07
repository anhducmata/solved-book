# Use the official Node.js 18 image as base
FROM node:18-alpine

# Install curl and postgresql-client for health checks and db connectivity checks
RUN apk add --no-cache curl postgresql-client

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (if available)
COPY package*.json ./

# Accept NODE_ENV as build argument (defaults to production)
ARG NODE_ENV=production

# Install dependencies based on environment
RUN if [ "$NODE_ENV" = "development" ] ; then npm ci ; else npm ci --only=production ; fi

# Copy the rest of the application code
COPY . .

# Copy and set permissions for entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create a non-root user to run the application
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

# Change ownership of the app directory to the nodejs user
RUN chown -R nodejs:nodejs /app
USER nodejs

# Expose the port the app runs on
EXPOSE 3000

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Use the entrypoint script
ENTRYPOINT ["docker-entrypoint.sh"]

# Run the application
CMD ["npm", "start"]
