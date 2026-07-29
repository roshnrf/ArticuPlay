"""enable row level security on all public tables

Revision ID: c8d61ef34346
Revises: 2a0405d370d8
Create Date: 2026-07-29 17:11:07.827812

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8d61ef34346'
down_revision: Union[str, None] = '2a0405d370d8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLES = [
    "users", "children", "drill_sessions", "drill_items",
    "phoneme_logs", "parent_tips", "word_content",
]


def upgrade() -> None:
    # Supabase exposes every public-schema table via its auto-generated REST/GraphQL
    # API by default, regardless of whether the app uses that API — RLS is opt-in per
    # table, not opt-out. This project only ever used Supabase as a direct Postgres
    # connection (this app's DB role is `postgres`, a superuser, which bypasses RLS —
    # so this migration has zero effect on the app itself), so the tables were left
    # reachable, unauthenticated, through Supabase's own API layer the whole time.
    # Enabling RLS with no policies = default-deny for every non-superuser role,
    # closing that off. If the anon/PostgREST API is ever deliberately used later,
    # add explicit policies then — don't remove this blanket enable.
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')


def downgrade() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')
