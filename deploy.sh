#!/bin/bash

# SolvedBook MCP Server Docker Deployment Script
# This script provides a comprehensive deployment workflow for the SolvedBook MCP Server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_PROJECT_NAME="solvedbook-mcp-server"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=2

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    log_info "Checking Docker availability..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Function to build images
build_images() {
    log_info "Building Docker images..."
    docker-compose build --no-cache
    log_success "Images built successfully"
}

# Function to start services
start_services() {
    local mode=$1
    log_info "Starting services in $mode mode..."
    
    if [ "$mode" = "production" ]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    log_success "Services started"
}

# Function to wait for health checks
wait_for_health() {
    log_info "Waiting for services to become healthy..."
    
    local retries=0
    while [ $retries -lt $HEALTH_CHECK_RETRIES ]; do
        if check_service_health; then
            log_success "All services are healthy"
            return 0
        fi
        
        retries=$((retries + 1))
        log_info "Health check attempt $retries/$HEALTH_CHECK_RETRIES"
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    log_error "Services failed to become healthy within timeout"
    return 1
}

# Function to check service health
check_service_health() {
    # Check if containers are running
    local running_containers=$(docker-compose ps -q | wc -l)
    if [ "$running_containers" -eq 0 ]; then
        return 1
    fi
    
    # Check database health
    if ! docker-compose exec -T postgres pg_isready -U solvedbook &> /dev/null; then
        return 1
    fi
    
    # Check application health
    if ! curl -sf http://localhost:3000/health &> /dev/null; then
        return 1
    fi
    
    return 0
}

# Function to show service status
show_status() {
    log_info "Service Status:"
    docker-compose ps
    
    echo ""
    log_info "Health Check Results:"
    
    # Database health
    if docker-compose exec -T postgres pg_isready -U solvedbook &> /dev/null; then
        log_success "Database: Healthy"
    else
        log_error "Database: Unhealthy"
    fi
    
    # Application health
    if curl -sf http://localhost:3000/health &> /dev/null; then
        local health_response=$(curl -s http://localhost:3000/health)
        log_success "Application: Healthy"
        echo "  Response: $health_response"
    else
        log_error "Application: Unhealthy"
    fi
}

# Function to show logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        log_info "Showing logs for all services..."
        docker-compose logs --tail=50
    else
        log_info "Showing logs for service: $service"
        docker-compose logs --tail=50 "$service"
    fi
}

# Function to stop services
stop_services() {
    log_info "Stopping services..."
    docker-compose down
    log_success "Services stopped"
}

# Function to clean up
cleanup() {
    log_info "Cleaning up containers, networks, and volumes..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    log_success "Cleanup completed"
}

# Function to backup database
backup_database() {
    local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    log_info "Creating database backup: $backup_file"
    
    docker-compose exec -T postgres pg_dump -U solvedbook solvedbook > "$backup_file"
    
    if [ -f "$backup_file" ] && [ -s "$backup_file" ]; then
        log_success "Database backup created: $backup_file"
    else
        log_error "Database backup failed"
        return 1
    fi
}

# Function to restore database
restore_database() {
    local backup_file=$1
    if [ -z "$backup_file" ]; then
        log_error "Please specify backup file to restore"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    log_info "Restoring database from: $backup_file"
    docker-compose exec -T postgres psql -U solvedbook -d solvedbook < "$backup_file"
    log_success "Database restored successfully"
}

# Function to show help
show_help() {
    cat << EOF
SolvedBook MCP Server Docker Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy [dev|prod]     Deploy the application (default: dev)
    start [dev|prod]      Start services (default: dev)
    stop                  Stop all services
    restart [dev|prod]    Restart services (default: dev)
    status                Show service status and health
    logs [service]        Show logs (all services or specific service)
    build                 Build Docker images
    cleanup               Clean up containers, networks, and volumes
    backup                Create database backup
    restore <file>        Restore database from backup file
    help                  Show this help message

Examples:
    $0 deploy prod        Deploy in production mode
    $0 start              Start in development mode
    $0 logs app           Show application logs
    $0 backup             Create database backup
    $0 restore backup_20250902_123456.sql

Options:
    --no-build           Skip building images during deploy
    --force              Force operation even if services are unhealthy

EOF
}

# Main deployment function
deploy() {
    local mode=${1:-dev}
    local skip_build=${2:-false}
    
    log_info "Starting deployment in $mode mode..."
    
    check_docker
    
    if [ "$skip_build" != "true" ]; then
        build_images
    fi
    
    start_services "$mode"
    
    if wait_for_health; then
        show_status
        log_success "Deployment completed successfully!"
        log_info "Application available at: http://localhost:3000"
        log_info "Health check: http://localhost:3000/health"
        if [ "$mode" = "dev" ]; then
            log_info "Database available at: localhost:5432"
        else
            log_info "Database available at: localhost:5433"
        fi
    else
        log_error "Deployment failed - services are not healthy"
        show_logs
        exit 1
    fi
}

# Parse command line arguments
case "${1:-help}" in
    deploy)
        mode=${2:-dev}
        skip_build=false
        if [[ "$@" == *"--no-build"* ]]; then
            skip_build=true
        fi
        deploy "$mode" "$skip_build"
        ;;
    start)
        check_docker
        start_services "${2:-dev}"
        wait_for_health
        show_status
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 2
        start_services "${2:-dev}"
        wait_for_health
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$2"
        ;;
    build)
        check_docker
        build_images
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        backup_database
        ;;
    restore)
        restore_database "$2"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
