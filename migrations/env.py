"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.database import Base
import app.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
      fileConfig(config.config_file_name)

target_metadata = Base.metadata

database_url = os.getenv("DATABASE_URL")
if not database_url:
      raise ValueError("DATABASE_URL environment variable not set")

config.set_main_option("sqlalchemy.url", database_url)


def run_migrations_offline() -> None:
      url = config.get_main_option("sqlalchemy.url")
      context.configure(
          url=url,
          target_metadata=target_metadata,
          literal_binds=True,
          dialect_opts={"paramstyle": "named"},
      )
      with context.begin_transaction():
                context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
      context.configure(connection=connection, target_metadata=target_metadata)
      with context.begin_transaction():
                context.run_migrations()


async def run_migrations_online() -> None:
      connectable = create_async_engine(
                config.get_main_option("sqlalchemy.url"),
                poolclass=pool.NullPool,
      )
      async with connectable.begin() as connection:
                await connection.run_sync(do_run_migrations)
            await connectable.dispose()


if context.is_offline_mode():
      run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
