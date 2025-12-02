"""Database connection management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import logging

from src.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)

# PostgreSQL engine
postgres_engine = create_engine(
    f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
    f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.debug
)

PostgresSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=postgres_engine)
    logger.info("Database tables initialized")


@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    session = PostgresSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()



