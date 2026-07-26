from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.child import ChildCreate, ChildRead, ChildUpdate
from app.services.child_service import ChildService

router = APIRouter()


def _service(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)) -> ChildService:
    return ChildService(db, parent_id=user.id)


@router.get("/", response_model=list[ChildRead])
async def list_children(service: ChildService = Depends(_service)):
    return await service.list_all()


@router.post("/", response_model=ChildRead, status_code=status.HTTP_201_CREATED)
async def create_child(body: ChildCreate, service: ChildService = Depends(_service)):
    return await service.create(body)


@router.get("/{child_id}", response_model=ChildRead)
async def get_child(child_id: UUID, service: ChildService = Depends(_service)):
    return await service.get(child_id)


@router.patch("/{child_id}", response_model=ChildRead)
async def update_child(child_id: UUID, body: ChildUpdate, service: ChildService = Depends(_service)):
    return await service.update(child_id, body)


@router.delete("/{child_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_child(child_id: UUID, service: ChildService = Depends(_service)):
    await service.delete(child_id)
