"""enable rls on alembic_version table

Revision ID: 59148773e0ea
Revises: c8d61ef34346
Create Date: 2026-07-29 17:18:23.978746

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '59148773e0ea'
down_revision: Union[str, None] = 'c8d61ef34346'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Missed in c8d61ef34346 — alembic_version is Alembic's own migration-tracking
    # table (not one of our SQLAlchemy models), so it wasn't in that migration's
    # hardcoded table list. Supabase's scanner caught it as a separate public-schema
    # table exposed via PostgREST with no RLS. Also revoke anon/authenticated grants
    # (belt-and-suspenders — this table has no legitimate app-level reason to be
    # reachable by either role at all, migrations run as the superuser).
    op.execute('ALTER TABLE "alembic_version" ENABLE ROW LEVEL SECURITY')
    op.execute('REVOKE ALL ON TABLE "alembic_version" FROM anon')
    op.execute('REVOKE ALL ON TABLE "alembic_version" FROM authenticated')


def downgrade() -> None:
    op.execute('ALTER TABLE "alembic_version" DISABLE ROW LEVEL SECURITY')
