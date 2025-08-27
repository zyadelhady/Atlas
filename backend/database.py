from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from dotenv import load_dotenv
import os
import datetime
from typing import AsyncGenerator

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL)


class Base(DeclarativeBase):
    pass

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(index=True)
    prompt: Mapped[str]
    response: Mapped[str]
    timestamp: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def get_async_session_local():
    return async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncGenerator:
    AsyncSessionLocal = get_async_session_local()
    async with AsyncSessionLocal() as session:
        yield session