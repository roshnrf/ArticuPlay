from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.child import Child
from app.schemas.child import ChildCreate, ChildRead, ChildUpdate


class ChildService:
    """All reads/writes are scoped to the requesting parent — a child profile is never
    visible or editable by anyone else, including other parents."""

    def __init__(self, db: AsyncSession, parent_id: UUID):
        self.db = db
        self.parent_id = parent_id

    async def list_all(self) -> list[ChildRead]:
        result = await self.db.scalars(select(Child).where(Child.parent_id == self.parent_id))
        return [ChildRead.model_validate(c) for c in result.all()]

    async def create(self, body: ChildCreate) -> ChildRead:
        child = Child(**body.model_dump(), parent_id=self.parent_id)
        self.db.add(child)
        await self.db.commit()
        await self.db.refresh(child)
        return ChildRead.model_validate(child)

    async def _get_owned(self, child_id: UUID) -> Child:
        child = await self.db.get(Child, child_id)
        if not child or child.parent_id != self.parent_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Child not found")
        return child

    async def get(self, child_id: UUID) -> ChildRead:
        return ChildRead.model_validate(await self._get_owned(child_id))

    async def update(self, child_id: UUID, body: ChildUpdate) -> ChildRead:
        child = await self._get_owned(child_id)
        for field, value in body.model_dump(exclude_none=True).items():
            setattr(child, field, value)
        await self.db.commit()
        await self.db.refresh(child)
        return ChildRead.model_validate(child)

    async def delete(self, child_id: UUID) -> None:
        child = await self._get_owned(child_id)
        await self.db.delete(child)
        await self.db.commit()
