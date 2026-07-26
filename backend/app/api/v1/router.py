from fastapi import APIRouter

from app.api.v1.endpoints import auth, children, sessions, asr, tts, content

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(children.router, prefix="/children", tags=["children"])
api_router.include_router(sessions.router, prefix="/session", tags=["session"])
api_router.include_router(asr.router, prefix="/session", tags=["session"])
api_router.include_router(tts.router, prefix="/session", tags=["session"])
api_router.include_router(content.router, prefix="/content", tags=["content"])
