# Kubernetes Deployment

Kubernetes manifests for deploying the Virality Engine.

## Files

- `deployment.yaml` - Standard Kubernetes Deployment
- `rollout.yaml` - Argo Rollouts for canary deployments
- `service.yaml` - Kubernetes Service
- `ingress.yaml` - Ingress with TLS 1.3
- `configmap.yaml` - Configuration
- `secret-template.yaml` - Secret template (DO NOT commit actual secrets)
- `hpa.yaml` - Horizontal Pod Autoscaler
- `pdb.yaml` - Pod Disruption Budget

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Docker registry for images
- NGINX Ingress Controller

## Quick Start

### Standard Deployment

```bash
# Create namespace
kubectl create namespace virality-production

# Create secrets
kubectl create secret generic virality-secrets \
  --from-literal=postgres-password='your-password' \
  --from-literal=clickhouse-password='your-password' \
  -n virality-production

# Deploy
kubectl apply -f configmap.yaml -n virality-production
kubectl apply -f service.yaml -n virality-production
kubectl apply -f deployment.yaml -n virality-production
kubectl apply -f hpa.yaml -n virality-production
kubectl apply -f pdb.yaml -n virality-production
kubectl apply -f ingress.yaml -n virality-production
```

### Canary Deployment (Argo Rollouts)

```bash
# Install Argo Rollouts
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# Deploy
kubectl apply -f configmap.yaml -n virality-production
kubectl apply -f service.yaml -n virality-production
kubectl apply -f rollout.yaml -n virality-production
kubectl apply -f hpa.yaml -n virality-production
kubectl apply -f pdb.yaml -n virality-production
kubectl apply -f ingress.yaml -n virality-production

# Monitor
kubectl argo rollouts get rollout virality-engine-api -n virality-production
```

## TLS Setup

See `tls-setup.md` for TLS 1.3 configuration instructions.

## Troubleshooting

### Check Status
```bash
kubectl get deployments -n virality-production
kubectl get pods -n virality-production
kubectl logs -f deployment/virality-engine-api -n virality-production
```

### Check Ingress
```bash
kubectl get ingress -n virality-production
kubectl describe ingress virality-engine-api-ingress -n virality-production
```

### Port Forward
```bash
kubectl port-forward svc/virality-engine-api 8000:8000 -n virality-production
```

## Security Notes

1. Never commit secrets to Git
2. Use sealed-secrets or external-secrets for secrets management
3. Configure TLS certificates for ingress
4. Use RBAC to restrict access
5. Enable network policies
