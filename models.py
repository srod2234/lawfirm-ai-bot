# models.py

import os
from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, Relationship, create_engine

# Use Postgres in prod; else fall back to a local SQLite file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local.db")
engine = create_engine(DATABASE_URL, echo=True)

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id")
    label: str
    file_path: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to pages
    pages: List["Page"] = Relationship(back_populates="document")

class Page(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    page_number: int
    text: str
    is_scanned: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="pages")

class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: int = Field(foreign_key="document.id")
    question: str
    answer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

def init_db():
    SQLModel.metadata.create_all(engine)
