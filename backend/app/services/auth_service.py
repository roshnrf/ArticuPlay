from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import Token, UserCreate, UserRead


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def register(self, body: UserCreate) -> UserRead:
        existing = await self.db.scalar(select(User).where(User.email == body.email))
        if existing:
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
        user = User(
            email=body.email,
            hashed_password=hash_password(body.password),
            full_name=body.full_name,
            role=body.role,
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return UserRead.model_validate(user)

    async def login(self, email: str, password: str) -> Token:
        user = await self.db.scalar(select(User).where(User.email == email))
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
        token = create_access_token(user.id)
        return Token(access_token=token)
