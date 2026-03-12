from sqlalchemy import Column, Integer, String, Date, ForeignKey
from database import Base

class HCP(Base):

    __tablename__ = "hcp"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String)

    specialization = Column(String)

    hospital = Column(String)

    city = Column(String)


class Interaction(Base):

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)

    hcp_id = Column(Integer, ForeignKey("hcp.id"))

    interaction_type = Column(String)

    product = Column(String)

    notes = Column(String)

    date = Column(String)

    follow_up = Column(String)