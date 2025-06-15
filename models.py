import os
from datetime import datetime
from typing import Optional, List

from sqlmodel import SQLModel, Field, create_engine, Relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local.db")
engine = create_engine(DATABASE_URL, echo=True)

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default="standard")  # "standard" or "admin"

    documents: List["Document"] = Relationship(back_populates="owner")
    messages: List["ChatMessage"] = Relationship(back_populates="user")

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="user.id")
    label: str
    file_path: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    owner: User = Relationship(back_populates="documents")
    pages: List["Page"] = Relationship(back_populates="document")
    messages: List["ChatMessage"] = Relationship(back_populates="document")

class Page(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    page_number: int
    text: str
    is_scanned: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Document = Relationship(back_populates="pages")

class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: int = Field(foreign_key="document.id")
    user_id: int = Field(foreign_key="user.id")
    question: str
    answer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    document: Document = Relationship(back_populates="messages")
    user: User = Relationship(back_populates="messages")

def init_db():
    SQLModel.metadata.create_all(engine)
