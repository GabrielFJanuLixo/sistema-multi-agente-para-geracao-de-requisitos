"""Add max_rounds and max_interviews to conversations."""

import sqlalchemy as sa
from alembic import op

revision = '0007'
down_revision = '0006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('conversations') as batch_op:
        batch_op.add_column(
            sa.Column(
                'max_rounds',
                sa.Integer(),
                nullable=False,
                server_default='2',
            )
        )
        batch_op.add_column(
            sa.Column(
                'max_interviews',
                sa.Integer(),
                nullable=False,
                server_default='2',
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('conversations') as batch_op:
        batch_op.drop_column('max_interviews')
        batch_op.drop_column('max_rounds')