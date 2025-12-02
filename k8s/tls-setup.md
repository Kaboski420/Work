# TLS 1.3 Setup Guide

Instructions for configuring TLS 1.3 for the Virality Engine API.

## Prerequisites

- NGINX Ingress Controller installed
- Domain name configured
- DNS pointing to ingress controller

## Setup Methods

### Method 1: cert-manager with Let's Encrypt (Recommended)

1. Install cert-manager:
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

2. Create ClusterIssuer:
```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

3. Create Certificate:
```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: virality-engine-tls
  namespace: virality-production
spec:
  secretName: virality-engine-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.virality-engine.example.com
```

4. Apply ingress:
```bash
kubectl apply -f k8s/ingress.yaml
```

### Method 2: Existing Certificate

```bash
kubectl create secret tls virality-engine-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  --namespace=virality-production
```

### Method 3: Self-Signed (Testing Only)

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=api.virality-engine.example.com"

kubectl create secret tls virality-engine-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=virality-production
```

## Verification

### Check TLS Secret
```bash
kubectl get secret virality-engine-tls -n virality-production
```

### Test TLS Connection
```bash
openssl s_client -connect api.virality-engine.example.com:443 -tls1_3
```

Expected output:
```
Protocol  : TLSv1.3
Cipher    : TLS_AES_256_GCM_SHA384
```

### Test with curl
```bash
curl -v https://api.virality-engine.example.com/health
```

## Troubleshooting

### Certificate Not Found
```bash
kubectl get secret virality-engine-tls -n virality-production
```

### TLS 1.3 Not Working
Check NGINX Ingress Controller version:
```bash
kubectl exec -n ingress-nginx <ingress-pod> -- nginx -V
```

## Security Best Practices

1. Use cert-manager for automatic renewal
2. Use Let's Encrypt for trusted certificates
3. Rotate certificates before expiration
4. Use strong private keys (RSA 2048+ or ECDSA P-256+)
