# Kosmos AI Scientist - Northflank Deployment Guide

This guide explains how to deploy Kosmos AI Scientist to [Northflank](https://northflank.com/) using the automated deployment template.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Template Structure](#template-structure)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying to Northflank, ensure you have:

1. **Northflank Account**: Sign up at [northflank.com](https://northflank.com/)
2. **Anthropic API Key**: Get from [console.anthropic.com](https://console.anthropic.com/)
3. **GitHub Repository**: Fork or clone the Kosmos repository

## Quick Start

### Option 1: One-Click Deploy (Recommended)

1. Click the deploy button below:

   [![Deploy to Northflank](https://img.shields.io/badge/Deploy%20to-Northflank-blue?style=for-the-badge&logo=northflank)](https://app.northflank.com/s/account/templates/new?template=https://raw.githubusercontent.com/jimmc414/Kosmos/master/northflank.json)

2. Fill in the required arguments:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `OPENAI_API_KEY`: (Optional) OpenAI API key for alternative provider

3. Click "Create" to start the deployment

4. Wait for the build and deployment to complete (~5-10 minutes)

### Option 2: CLI Deployment

```bash
# Install Northflank CLI
npm install -g @northflank/cli

# Login to Northflank
northflank login

# Create project from template
northflank template create \
  --file northflank.json \
  --project kosmos-prod \
  --arguments ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Template Structure

The `northflank.json` template creates the following resources:

### Services

| Service | Description | Plan |
|---------|-------------|------|
| `kosmos-app` | Main application server | nf-compute-200 |
| `kosmos-build` | Build service for Docker images | nf-compute-200 |

### Addons

| Addon | Type | Description |
|-------|------|-------------|
| `kosmos-postgres` | PostgreSQL 15 | Primary database |
| `kosmos-redis` | Redis 7 | Caching layer |

### Jobs

| Job | Description | Trigger |
|-----|-------------|---------|
| `kosmos-db-migrate` | Database migrations | Manual |

### Secrets

| Secret Group | Variables |
|--------------|-----------|
| `kosmos-secrets` | ANTHROPIC_API_KEY, OPENAI_API_KEY, NEO4J_PASSWORD |

## Manual Deployment

If you prefer to set up resources manually:

### 1. Create Project

```bash
northflank project create kosmos-prod
```

### 2. Create PostgreSQL Addon

In the Northflank dashboard:
1. Go to your project
2. Click "Add Addon"
3. Select PostgreSQL
4. Choose version 15
5. Configure:
   - Name: `kosmos-postgres`
   - Plan: nf-compute-50
   - Storage: 10GB

### 3. Create Redis Addon

1. Click "Add Addon"
2. Select Redis
3. Choose version 7
4. Configure:
   - Name: `kosmos-redis`
   - Plan: nf-compute-20
   - Storage: 1GB

### 4. Create Build Service

1. Click "Add Service" → "Build Service"
2. Connect your GitHub repository
3. Configure:
   - Name: `kosmos-build`
   - Dockerfile path: `/Dockerfile`
   - Build context: `/`

### 5. Create Deployment Service

1. Click "Add Service" → "Deployment Service"
2. Select your build service as source
3. Configure:
   - Name: `kosmos-app`
   - Port: 8000
   - Health check path: `/health`

### 6. Configure Environment Variables

Add the following environment variables to your deployment:

```
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Database (auto-configured from addon)
DATABASE_URL=${KOSMOS_POSTGRES_CONNECTION_STRING}

# Redis (auto-configured from addon)
REDIS_ENABLED=true
REDIS_URL=${KOSMOS_REDIS_CONNECTION_STRING}

# Application
LOG_LEVEL=INFO
CLAUDE_MODEL=claude-sonnet-4-20250514
ENABLE_CONCURRENT_OPERATIONS=true
MAX_CONCURRENT_EXPERIMENTS=4
```

### 7. Run Database Migrations

Create a job to run migrations:

```bash
alembic upgrade head
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key for Claude |
| `OPENAI_API_KEY` | No | - | OpenAI API key (alternative provider) |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `REDIS_ENABLED` | No | `true` | Enable Redis caching |
| `REDIS_URL` | No | - | Redis connection string |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-20250514` | Claude model to use |
| `MAX_CONCURRENT_EXPERIMENTS` | No | `4` | Max parallel experiments |

### Scaling

To scale your deployment:

1. **Horizontal Scaling**: Increase instances in the deployment settings
2. **Vertical Scaling**: Upgrade to a larger compute plan

Recommended configurations:

| Use Case | Instances | Plan | Memory |
|----------|-----------|------|--------|
| Development | 1 | nf-compute-50 | 512MB |
| Production | 2 | nf-compute-200 | 2GB |
| High-throughput | 4+ | nf-compute-400 | 4GB |

### Custom Domains

1. Go to your deployment → Networking
2. Add your custom domain
3. Configure DNS:
   - CNAME record pointing to your Northflank domain

## Monitoring

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Liveness probe | `{"status": "healthy"}` |
| `/health/ready` | Readiness probe | `{"status": "ready"}` |
| `/health/metrics` | System metrics | CPU, memory, disk usage |

### Logs

View logs in the Northflank dashboard:
1. Go to your deployment
2. Click "Logs" tab
3. Filter by severity or search

### Metrics

Northflank provides built-in metrics:
- CPU usage
- Memory usage
- Network I/O
- Request count
- Response times

## Troubleshooting

### Common Issues

#### Build Fails

```
Error: Docker build failed
```

**Solution**: Check Dockerfile syntax and ensure all dependencies are available.

```bash
# Verify locally
docker build -t kosmos:test .
```

#### Database Connection Error

```
Error: Cannot connect to PostgreSQL
```

**Solution**: 
1. Verify the addon is running
2. Check connection string format
3. Ensure the addon is linked to your deployment

#### Health Check Fails

```
Error: Health check failed - /health returned 5xx
```

**Solution**:
1. Check application logs for startup errors
2. Verify all environment variables are set
3. Ensure database migrations have run

### Debug Mode

Enable debug logging:

```
LOG_LEVEL=DEBUG
DEBUG=true
```

### Support

- **Northflank Docs**: [docs.northflank.com](https://docs.northflank.com/)
- **Kosmos Issues**: [GitHub Issues](https://github.com/jimmc414/Kosmos/issues)
- **Northflank Support**: [support.northflank.com](https://support.northflank.com/)

## Security Considerations

1. **API Keys**: Always use secret groups for sensitive data
2. **TLS**: Northflank provides automatic TLS for all deployments
3. **Network Isolation**: Addons are only accessible within your project
4. **Access Control**: Use Northflank teams for role-based access

## Cost Optimization

- Use smaller compute plans for development
- Enable auto-sleep for non-production environments
- Monitor resource usage and right-size deployments
- Use Redis caching to reduce API calls

---

## Next Steps

After successful deployment:

1. **Verify**: Access your deployment URL and check `/health`
2. **Configure**: Set up custom domains if needed
3. **Monitor**: Set up alerts for health check failures
4. **Integrate**: Connect your CI/CD pipeline for automatic deployments

For more information, see the main [Deployment Guide](./deployment-guide.md).

