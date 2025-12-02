"""
Keycloak Authentication Middleware for FastAPI.

Implements RBAC authentication and authorization using Keycloak.
"""

import logging
from typing import Optional, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from datetime import datetime

from src.config import settings

logger = logging.getLogger(__name__)

# Try to import jose
try:
    from jose import jwt, JWTError
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    jwt = None
    JWTError = Exception
    logger.warning("python-jose not available. JWT verification will be limited.")

# HTTP Bearer token scheme
security = HTTPBearer()


class KeycloakAuth:
    """Keycloak authentication and authorization service."""
    
    def __init__(self):
        self.keycloak_url = settings.keycloak_url
        self.realm = settings.keycloak_realm
        self.client_id = "virality-engine"  # Default client ID
        self.public_key_cache = None
        self.public_key_cache_expiry = None
    
    async def get_public_key(self) -> str:
        """
        Get Keycloak realm public key for JWT verification.
        
        Returns:
            Public key in PEM format
        """
        try:
            # Cache public key for 1 hour
            if self.public_key_cache and self.public_key_cache_expiry:
                if datetime.utcnow() < self.public_key_cache_expiry:
                    return self.public_key_cache
            
            # Fetch public key from Keycloak
            url = f"{self.keycloak_url}/realms/{self.realm}"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                realm_info = response.json()
                
                # Extract public key
                public_key = realm_info.get("public_key")
                if not public_key:
                    raise ValueError("Public key not found in realm info")
                
                # Cache for 1 hour
                self.public_key_cache = public_key
                self.public_key_cache_expiry = datetime.utcnow().replace(
                    hour=datetime.utcnow().hour + 1
                )
                
                return public_key
                
        except Exception as e:
            logger.error(f"Error fetching Keycloak public key: {e}")
            raise HTTPException(
                status_code=503,
                detail="Authentication service unavailable"
            )
    
    async def verify_token(self, token: str) -> dict:
        """
        Verify JWT token with Keycloak.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded token payload
        """
        if not JOSE_AVAILABLE:
            # Fallback: basic token validation (development only)
            logger.warning("JWT verification not available. Using basic validation.")
            if settings.environment == "development":
                # In development, accept any token format
                try:
                    # Try to decode without verification
                    decoded = jwt.decode(token, options={"verify_signature": False})
                    return decoded
                except:
                    raise HTTPException(status_code=401, detail="Invalid token format")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Authentication service unavailable"
                )
        
        try:
            # Get public key
            public_key = await self.get_public_key()
            
            # Decode and verify token
            # Note: In production, use proper JWK format conversion
            # For now, we'll use a simplified approach
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.client_id,
                options={"verify_signature": True}
            )
            
            return decoded
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            raise HTTPException(
                status_code=401,
                detail="Token verification failed"
            )
    
    async def get_user_roles(self, token_payload: dict) -> List[str]:
        """
        Extract user roles from token payload.
        
        Args:
            token_payload: Decoded JWT token
        
        Returns:
            List of role names
        """
        # Keycloak roles are typically in 'realm_access' or 'resource_access'
        roles = []
        
        # Realm roles
        if "realm_access" in token_payload:
            roles.extend(token_payload["realm_access"].get("roles", []))
        
        # Client roles
        if "resource_access" in token_payload:
            client_access = token_payload["resource_access"].get(self.client_id, {})
            roles.extend(client_access.get("roles", []))
        
        return roles
    
    def check_role(self, user_roles: List[str], required_roles: List[str]) -> bool:
        """
        Check if user has any of the required roles.
        
        Args:
            user_roles: User's roles
            required_roles: Required roles (any match)
        
        Returns:
            True if user has at least one required role
        """
        if not required_roles:
            return True  # No role requirement
        
        return any(role in user_roles for role in required_roles)


# Global Keycloak auth instance
keycloak_auth = KeycloakAuth()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """
    Dependency to get current authenticated user.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user": user}
    """
    token = credentials.credentials
    token_payload = await keycloak_auth.verify_token(token)
    
    return {
        "sub": token_payload.get("sub"),
        "email": token_payload.get("email"),
        "username": token_payload.get("preferred_username"),
        "roles": await keycloak_auth.get_user_roles(token_payload),
        "token_payload": token_payload
    }


def require_roles(required_roles: List[str]):
    """
    Dependency factory to require specific roles.
    
    Usage:
        @app.get("/admin")
        async def admin_route(
            user: dict = Depends(require_roles(["admin", "moderator"]))
        ):
            return {"message": "Admin access"}
    """
    async def role_checker(user: dict = Depends(get_current_user)) -> dict:
        user_roles = user.get("roles", [])
        if not keycloak_auth.check_role(user_roles, required_roles):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {required_roles}"
            )
        return user
    
    return role_checker


# Optional: Allow bypassing auth in development
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(
        HTTPBearer(auto_error=False)
    )
) -> Optional[dict]:
    """
    Optional authentication - allows unauthenticated access in development.
    
    Usage:
        @app.get("/public")
        async def public_route(user: Optional[dict] = Depends(get_optional_user)):
            if user:
                return {"authenticated": True, "user": user}
            return {"authenticated": False}
    """
    if not credentials:
        if settings.environment == "development":
            return None  # Allow unauthenticated in dev
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        token = credentials.credentials
        token_payload = await keycloak_auth.verify_token(token)
        return {
            "sub": token_payload.get("sub"),
            "email": token_payload.get("email"),
            "username": token_payload.get("preferred_username"),
            "roles": await keycloak_auth.get_user_roles(token_payload),
        }
    except HTTPException:
        if settings.environment == "development":
            return None
        raise

