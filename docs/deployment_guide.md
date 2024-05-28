# Deployment Guide

## Overview

This guide covers deploying the Predictive Workplace Safety Analytics Platform in production environments. The platform supports multiple deployment options including Docker containers, Kubernetes clusters, and cloud platforms.

## Prerequisites

### System Requirements

**Minimum Hardware:**
- CPU: 4 cores (8 recommended)
- RAM: 16GB (32GB recommended)
- Storage: 100GB SSD (500GB recommended)
- Network: 1Gbps connection

**Software Requirements:**
- Docker 20.10+
- Docker Compose 2.0+
- PostgreSQL 14+
- Python 3.8+
- Git

### Cloud Platform Requirements

**AWS:**
- EC2 instance: t3.xlarge or larger
- RDS PostgreSQL instance
- S3 bucket for data storage
- IAM roles with appropriate permissions

**Azure:**
- Virtual Machine: Standard_D4s_v3 or larger
- Azure Database for PostgreSQL
- Storage Account
- Service Principal with required permissions

**Google Cloud:**
- Compute Engine: n2-standard-4 or larger
- Cloud SQL for PostgreSQL
- Cloud Storage bucket
- Service Account with required permissions

## Quick Start Deployment

### 1. Docker Compose (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/pranav-reveendran/Predictive-Workplace-Safety-Analytics-Platform.git
cd Predictive-Workplace-Safety-Analytics-Platform

# Copy and configure environment
cp config/config.template.yaml config/config.yaml
# Edit config.yaml with your settings

# Start all services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

### 2. Manual Installation

```bash
# Install Python dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup database
python src/database/init_db.py

# Run setup script
./scripts/setup.sh

# Start the application
python src/app.py
```

## Production Deployment

### Docker Production Setup

#### 1. Build Production Image

```dockerfile
# Create production Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY sql/ ./sql/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "src/app.py"]
```

#### 2. Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/safety_analytics
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
      - DEBUG=false
    depends_on:
      - db
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=safety_analytics
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/create_schema.sql:/docker-entrypoint-initdb.d/create_schema.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
```

### Kubernetes Deployment

#### 1. Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: safety-analytics

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: safety-config
  namespace: safety-analytics
data:
  config.yaml: |
    database:
      host: postgres-service
      port: 5432
      name: safety_analytics
      user: postgres
    
    model:
      ensemble_method: stacking
      target_accuracy: 0.84
      
    api:
      host: 0.0.0.0
      port: 8000
```

#### 2. Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: safety-analytics
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: safety_analytics
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: safety-analytics
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

#### 3. Application Deployment

```yaml
# k8s/app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-analytics
  namespace: safety-analytics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: safety-analytics
  template:
    metadata:
      labels:
        app: safety-analytics
    spec:
      containers:
      - name: safety-analytics
        image: your-registry/safety-analytics:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: safety-config

---
apiVersion: v1
kind: Service
metadata:
  name: safety-analytics-service
  namespace: safety-analytics
spec:
  selector:
    app: safety-analytics
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Cloud Platform Deployments

#### AWS ECS with Fargate

```json
{
  "family": "safety-analytics",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "safety-analytics",
      "image": "your-account.dkr.ecr.region.amazonaws.com/safety-analytics:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/safety_analytics"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/safety-analytics",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Azure Container Instances

```yaml
# azure-deployment.yaml
apiVersion: 2018-10-01
location: eastus
name: safety-analytics-group
properties:
  containers:
  - name: safety-analytics
    properties:
      image: your-registry.azurecr.io/safety-analytics:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 2
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: DATABASE_URL
        value: postgresql://user:pass@postgres-server:5432/safety_analytics
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
  restartPolicy: Always
```

## Configuration Management

### Environment Variables

```bash
# Required environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/safety_analytics"
export SECRET_KEY="your-secret-key-here"
export REDIS_URL="redis://localhost:6379"
export MODEL_VERSION="v2.1.0"
export LOG_LEVEL="INFO"
export DEBUG="false"

# Optional environment variables
export SENTRY_DSN="https://your-sentry-dsn"
export NEW_RELIC_LICENSE_KEY="your-newrelic-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
```

### Configuration Files

```yaml
# config/production.yaml
database:
  host: ${DATABASE_HOST}
  port: 5432
  name: safety_analytics
  user: ${DATABASE_USER}
  password: ${DATABASE_PASSWORD}
  pool_size: 20
  max_overflow: 30

redis:
  url: ${REDIS_URL}
  connection_pool_max_connections: 20

model:
  ensemble_method: stacking
  target_accuracy: 0.84
  batch_size: 1000
  prediction_cache_ttl: 3600

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  worker_class: gunicorn.workers.ggevent.GeventWorker
  max_requests: 1000
  max_requests_jitter: 100

logging:
  level: INFO
  format: json
  handlers:
    - console
    - file
    - sentry

monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_interval: 30
```

## Security Configuration

### SSL/TLS Setup

```nginx
# nginx.conf
server {
    listen 80;
    server_name api.workplace-safety-analytics.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.workplace-safety-analytics.com;

    ssl_certificate /etc/ssl/certs/workplace-safety.crt;
    ssl_certificate_key /etc/ssl/private/workplace-safety.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_dhparam /etc/ssl/certs/dhparam.pem;

    location / {
        proxy_pass http://safety-analytics:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Rules

```bash
# UFW firewall rules
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow from 10.0.0.0/8 to any port 5432  # Database access
ufw enable
```

## Monitoring and Observability

### Health Check Endpoints

```python
# src/health.py
@app.route('/health')
def health_check():
    """Basic health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': app.config['VERSION']
    })

@app.route('/ready')
def readiness_check():
    """Readiness check with dependencies"""
    checks = {
        'database': check_database_connection(),
        'redis': check_redis_connection(),
        'model': check_model_availability()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return jsonify({
        'status': 'ready' if all_healthy else 'not_ready',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }), status_code
```

### Prometheus Metrics

```python
# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
prediction_requests = Counter('prediction_requests_total', 'Total prediction requests')
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
active_connections = Gauge('active_database_connections', 'Active DB connections')

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
```

### Logging Configuration

```python
# src/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure structured logging"""
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )
```

## Backup and Recovery

### Database Backup

```bash
#!/bin/bash
# scripts/backup_database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/database"
DB_NAME="safety_analytics"
DB_USER="postgres"
DB_HOST="localhost"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 30 backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: backup_$DATE.sql.gz"
```

### Automated Backup with Cron

```bash
# Add to crontab: crontab -e
# Daily backup at 2 AM
0 2 * * * /path/to/scripts/backup_database.sh

# Weekly full backup
0 1 * * 0 /path/to/scripts/full_backup.sh
```

## Performance Optimization

### Database Optimization

```sql
-- Performance tuning settings
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

SELECT pg_reload_conf();
```

### Application Tuning

```python
# src/config.py
class ProductionConfig:
    """Production configuration"""
    
    # Database connection pooling
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30
    }
    
    # Redis configuration
    REDIS_CONNECTION_POOL_KWARGS = {
        'max_connections': 20,
        'retry_on_timeout': True
    }
    
    # Gunicorn settings
    WORKERS = 4
    WORKER_CLASS = 'gevent'
    WORKER_CONNECTIONS = 1000
    MAX_REQUESTS = 1000
    MAX_REQUESTS_JITTER = 100
```

## Troubleshooting

### Common Issues

**Database Connection Issues:**
```bash
# Check database connectivity
pg_isready -h localhost -p 5432 -U postgres

# Check connection limits
SELECT count(*) FROM pg_stat_activity;
```

**Memory Issues:**
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Check application memory
docker stats safety-analytics
```

**Performance Issues:**
```bash
# Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
WHERE mean_time > 1000 
ORDER BY mean_time DESC;

# Monitor API response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"
```

### Log Analysis

```bash
# Application logs
docker logs safety-analytics --tail 100 -f

# Database logs
tail -f /var/log/postgresql/postgresql-14-main.log

# System logs
journalctl -u safety-analytics -f
```

This deployment guide provides comprehensive instructions for deploying the platform in various environments while ensuring security, scalability, and reliability. 