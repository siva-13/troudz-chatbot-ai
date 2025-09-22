#!/bin/bash
# Complete RAG Application Deployment Script
# Clones from GitHub, installs dependencies, sets up nginx, and runs the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="https://github.com/siva-13/troudz-chatbot-ai.git"
BRANCH="main"
APP_DIR="/var/www/rag-app"
APP_USER="www-data"
SERVICE_NAME="rag-app"
DOMAIN=""

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        error "Please run this script as root (use sudo)"
    fi
}

# Get user input for configuration
get_configuration() {
    log "📋 Configuration Setup"
    echo "================================"

    # Domain configuration
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
    if [ -z "$DOMAIN" ]; then
        error "Domain name is required"
    fi

    # API Keys
    read -p "Enter your OpenAI API Key: " OPENAI_KEY
    if [ -z "$OPENAI_KEY" ]; then
        error "OpenAI API Key is required"
    fi

    read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY
    if [ -z "$AWS_ACCESS_KEY" ]; then
        warning "AWS Access Key not provided. Application will use local storage."
    fi

    if [ ! -z "$AWS_ACCESS_KEY" ]; then
        read -p "Enter your AWS Secret Access Key: " AWS_SECRET_KEY
        read -p "Enter AWS Region (default: us-east-1): " AWS_REGION
        AWS_REGION=${AWS_REGION:-ap-southeast-2}
    fi

    # Confirmation
    echo ""
    log "📋 Configuration Summary:"
    echo "Domain: $DOMAIN"
    echo "OpenAI API Key: ${OPENAI_KEY:0:8}..."
    if [ ! -z "$AWS_ACCESS_KEY" ]; then
        echo "AWS Access Key: ${AWS_ACCESS_KEY:0:8}..."
        echo "AWS Region: $AWS_REGION"
    else
        echo "Storage: Local mode (no AWS)"
    fi
    echo ""
    read -p "Proceed with deployment? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        error "Deployment cancelled"
    fi
}

# Update system packages
update_system() {
    log "📦 Updating system packages..."
    apt update && apt upgrade -y
    apt install -y git curl wget software-properties-common
}

# Install Python and dependencies
install_python() {
    log "🐍 Installing Python and pip..."
    apt install -y python3 python3-pip python3-venv python3-dev

    # Install system dependencies for Python packages
    apt install -y build-essential libssl-dev libffi-dev python3-setuptools

    log "✅ Python installation completed"
}

# Install Nginx
install_nginx() {
    log "🌐 Installing Nginx..."
    apt install -y nginx

    # Start and enable nginx
    systemctl start nginx
    systemctl enable nginx

    log "✅ Nginx installed and started"
}

# Install Certbot for SSL
install_certbot() {
    log "🔒 Installing Certbot for SSL certificates..."
    apt install -y certbot python3-certbot-nginx
    log "✅ Certbot installed"
}

# Clone repository
clone_repository() {
    log "📥 Cloning repository from GitHub..."

    # Remove existing directory if it exists
    if [ -d "$APP_DIR" ]; then
        warning "Directory $APP_DIR already exists. Removing..."
        rm -rf "$APP_DIR"
    fi

    # Clone repository
    git clone -b "$BRANCH" "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"

    log "✅ Repository cloned successfully"
}

# Set up Python environment
setup_python_environment() {
    log "🔧 Setting up Python virtual environment..."
    cd "$APP_DIR"

    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    if [ -f "requirements_final_.txt" ]; then
        log "Installing from requirements_final_fix.txt..."
        pip install -r requirements_final.txt
    elif [ -f "requirements.txt" ]; then
        log "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        warning "No requirements file found. Installing basic dependencies..."
        pip install fastapi uvicorn[standard] python-docx openai faiss-cpu boto3 python-multipart pydantic tiktoken numpy packaging httpx
    fi

    log "✅ Python environment set up successfully"
}

# Create systemd service
create_service() {
    log "⚙️  Creating systemd service..."

    # Determine which server file to use
    if [ -z "$AWS_ACCESS_KEY" ]; then
        SERVER_FILE="fastapi_server_local.py"
        info "Using local server (no AWS credentials provided)"
    else
        SERVER_FILE="fastapi_server.py"
        info "Using AWS-enabled server"
    fi

    # Create service file
    cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=RAG Application FastAPI Server
After=network.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
Environment=OPENAI_API_KEY=$OPENAI_KEY
EOF

    # Add AWS credentials if provided
    if [ ! -z "$AWS_ACCESS_KEY" ]; then
        cat >> /etc/systemd/system/$SERVICE_NAME.service << EOF
Environment=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY
Environment=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_KEY
Environment=AWS_DEFAULT_REGION=$AWS_REGION
EOF
    fi

    # Complete service file
    cat >> /etc/systemd/system/$SERVICE_NAME.service << EOF
ExecStart=$APP_DIR/venv/bin/python -m uvicorn fastapi_server:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

    log "✅ Systemd service created"
}

# Configure Nginx
configure_nginx() {
    log "🌐 Configuring Nginx..."

    # Create nginx configuration
    cat > /etc/nginx/sites-available/rag-app << EOF
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;  # Replace with your domain

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        #add_header Access-Control-Allow-Origin "*";
        #add_header Access-Control-Allow-Origin "http://localhost:5173";
        #add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        #add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization" always;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF

    # Remove default site and enable our site
    rm -f /etc/nginx/sites-enabled/default
    ln -sf /etc/nginx/sites-available/rag-app /etc/nginx/sites-enabled/

    # Test nginx configuration
    nginx -t || error "Nginx configuration test failed"

    log "✅ Nginx configured successfully"
}

# Set up SSL certificate
setup_ssl() {
    log "🔒 Setting up SSL certificate..."

    # First, reload nginx to serve HTTP temporarily
    systemctl reload nginx

    # Get SSL certificate
    if certbot --nginx -d "$DOMAIN" -d "www.$DOMAIN" --non-interactive --agree-tos --email "admin@$DOMAIN" --redirect; then
        log "✅ SSL certificate obtained and configured"
    else
        warning "SSL certificate setup failed. The site will be accessible via HTTP for now."
        warning "You can manually run: certbot --nginx -d $DOMAIN"
    fi
}

# Set permissions
set_permissions() {
    log "🔒 Setting correct permissions..."

    # Change ownership to www-data
    chown -R $APP_USER:$APP_USER "$APP_DIR"

    # Set appropriate permissions
    chmod -R 755 "$APP_DIR"

    # Ensure log directories exist
    mkdir -p "$APP_DIR/logs"
    chown -R $APP_USER:$APP_USER "$APP_DIR/logs"

    log "✅ Permissions set correctly"
}

# Configure firewall
configure_firewall() {
    log "🛡️  Configuring firewall..."

    # Install and configure UFW
    apt install -y ufw

    # Set up firewall rules
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 'Nginx Full'

    # Enable firewall (non-interactive)
    echo "y" | ufw enable

    log "✅ Firewall configured"
}

# Start services
start_services() {
    log "🚀 Starting services..."

    # Reload systemd and start services
    systemctl daemon-reload
    systemctl enable $SERVICE_NAME
    systemctl start $SERVICE_NAME
    systemctl reload nginx

    # Wait a moment for services to start
    sleep 3

    # Check service status
    if systemctl is-active --quiet $SERVICE_NAME; then
        log "✅ RAG application service started successfully"
    else
        error "Failed to start RAG application service"
    fi

    if systemctl is-active --quiet nginx; then
        log "✅ Nginx service is running"
    else
        error "Nginx service failed to start"
    fi
}

# Show deployment summary
show_summary() {
    log "🎉 Deployment completed successfully!"
    echo ""
    echo "================================================"
    echo "  RAG APPLICATION DEPLOYMENT SUMMARY"
    echo "================================================"
    echo ""
    echo "🌐 Application URL:"
    echo "   https://$DOMAIN"
    echo "   https://$DOMAIN/docs (API Documentation)"
    echo "   https://$DOMAIN/health (Health Check)"
    echo ""
    echo "📊 Service Management:"
    echo "   Status:  sudo systemctl status $SERVICE_NAME"
    echo "   Logs:    sudo journalctl -u $SERVICE_NAME -f"
    echo "   Restart: sudo systemctl restart $SERVICE_NAME"
    echo ""
    echo "🔧 Nginx Management:"
    echo "   Status:  sudo systemctl status nginx"
    echo "   Reload:  sudo systemctl reload nginx"
    echo "   Logs:    sudo tail -f /var/log/nginx/access.log"
    echo ""
    echo "📁 Application Directory: $APP_DIR"
    echo ""
    echo "🔑 API Keys configured:"
    echo "   OpenAI API Key: ✅"
    if [ ! -z "$AWS_ACCESS_KEY" ]; then
        echo "   AWS Credentials: ✅"
    else
        echo "   AWS Credentials: ❌ (Using local storage)"
    fi
    echo ""
    echo "🔒 SSL Certificate: $([ -f /etc/letsencrypt/live/$DOMAIN/fullchain.pem ] && echo "✅ Configured" || echo "⚠️  Manual setup required")"
    echo ""
    echo "================================================"
}

# Main deployment function
main() {
    log "🚀 Starting RAG Application Deployment"
    echo "========================================"
    echo ""

    # Pre-flight checks
    check_root

    # Get configuration from user
    get_configuration

    # Installation steps
    update_system
    install_python
    install_nginx
    install_certbot

    # Application setup
    clone_repository
    setup_python_environment
    create_service
    set_permissions

    # Server configuration
    configure_nginx
    configure_firewall

    # Start services
    start_services

    # Set up SSL (after services are running)
    setup_ssl

    # Show summary
    show_summary

    log "✅ Deployment script completed!"
}

# Run main function
main "$@"

