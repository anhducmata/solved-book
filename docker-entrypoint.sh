#!/bin/sh

# Docker entrypoint script for SolvedBook MCP Server
set -e

echo "Starting SolvedBook MCP Server..."
echo "Environment: $NODE_ENV"
echo "Database Host: $DB_HOST"
echo "Database Port: $DB_PORT"
echo "Application Port: $PORT"

# Wait for database to be ready
echo "Waiting for database connection..."
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"; do
  echo "Database is unavailable - sleeping"
  sleep 2
done

echo "Database is ready!"

# Start the application
exec "$@"
