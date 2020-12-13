import os
from contextlib import contextmanager

from sqlalchemy import (JSON, Boolean, CheckConstraint, Column, DateTime,
                        Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine, or_, text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def parse_pgpass_from_file():
    with open(os.path.expanduser('~/.pgpass'), 'r') as f:
        info = f.read().strip(' \t\n\r')
    return info


Base = declarative_base()
engine = create_engine(parse_pgpass_from_file())
Session = sessionmaker(bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


class Modelzoo(Base):
    __tablename__ = 'modelzoo'
    model_id = Column(Text, primary_key=True)
    created_on = Column(DateTime, nullable=False)
    model_task = Column(Text, nullable=False)
    model_name = Column(Text, nullable=False)
    expname = Column(Text, nullable=False)
    status = Column(Text)
    maintainer = Column(Text)
    dataset_name = Column(Text)
    split = Column(Text)
    score_name = Column(Text)
    score = Column(Float)
