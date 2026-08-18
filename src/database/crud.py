# src/database/crud.py

from sqlalchemy.orm import Session
from .models import Image, Detection, Quantification, ChatbotLog


def create_image(db: Session, filename: str):
    img = Image(filename=filename)
    db.add(img)
    db.commit()
    db.refresh(img)
    return img


def add_detection(db: Session, image_id, material, conf, bbox):
    det = Detection(
        image_id=image_id,
        material=material,
        confidence=conf,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3]
    )
    db.add(det)
    db.commit()


def add_quantification(
    db: Session,
    image_id,
    material,
    count=None,
    volume_m3=None,
    mass_kg=None
):
    q = Quantification(
        image_id=image_id,
        material=material,
        count=count,
        volume_m3=volume_m3,
        mass_kg=mass_kg
    )
    db.add(q)
    db.commit()


def log_chat(db: Session, query, response):
    log = ChatbotLog(query=query, response=response)
    db.add(log)
    db.commit()
