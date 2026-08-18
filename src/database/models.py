# src/database/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, server_default=func.now())

    detections = relationship("Detection", back_populates="image")
    quantification = relationship("Quantification", back_populates="image")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))

    material = Column(String)
    confidence = Column(Float)

    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)

    image = relationship("Image", back_populates="detections")


class Quantification(Base):
    __tablename__ = "quantification"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))

    material = Column(String)

    count = Column(Integer, nullable=True)
    volume_m3 = Column(Float, nullable=True)
    mass_kg = Column(Float, nullable=True)

    image = relationship("Image", back_populates="quantification")


class ChatbotLog(Base):
    __tablename__ = "chatbot_logs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    response = Column(String)
    timestamp = Column(DateTime, server_default=func.now())
