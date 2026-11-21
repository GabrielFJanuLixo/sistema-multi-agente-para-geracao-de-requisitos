"""Add role to agents and status/specification_json to conversations."""

import sqlalchemy as sa
from alembic import op

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    agent_role = sa.Enum(
        'INTERVIEWER', 'BACKEND', 'FRONTEND', 'QA', name='agentrole'
    )
    conversation_status = sa.Enum(
        'IDLE',
        'WAITING_USER',
        'COMPLETED',
        'COMPLETED_WITH_PENDING',
        name='conversationstatus',
    )

    with op.batch_alter_table('agents') as batch_op:
        batch_op.alter_column(
            'system_prompt',
            existing_type=sa.String(),
            type_=sa.Text(),
            existing_nullable=False,
        )
        batch_op.add_column(sa.Column('role', agent_role, nullable=True))
        batch_op.create_unique_constraint('uq_agents_role', ['role'])

    with op.batch_alter_table('conversations') as batch_op:
        batch_op.add_column(
            sa.Column(
                'status',
                conversation_status,
                nullable=False,
                server_default='IDLE',
            )
        )
        batch_op.add_column(
            sa.Column('specification_json', sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('conversations') as batch_op:
        batch_op.drop_column('specification_json')
        batch_op.drop_column('status')

    with op.batch_alter_table('agents') as batch_op:
        batch_op.drop_constraint('uq_agents_role', type_='unique')
        batch_op.drop_column('role')
        batch_op.alter_column(
            'system_prompt',
            existing_type=sa.Text(),
            type_=sa.String(),
            existing_nullable=False,
        )

    sa.Enum(name='conversationstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='agentrole').drop(op.get_bind(), checkfirst=True)