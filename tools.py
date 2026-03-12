from langchain.tools import tool
from database import SessionLocal
from models import HCP, Interaction


@tool
def search_hcp(name: str):

    db = SessionLocal()

    hcp = db.query(HCP).filter(HCP.name.ilike(f"%{name}%")).first()

    if hcp:

        return f"HCP found: {hcp.name}, {hcp.specialization}, {hcp.hospital}"

    return "HCP not found"



@tool
def log_interaction(
    hcp_name: str,
    interaction_type: str,
    product: str,
    notes: str,
    date: str,
    follow_up: str
):

    db = SessionLocal()

    hcp = db.query(HCP).filter(HCP.name == hcp_name).first()

    if not hcp:

        hcp = HCP(name=hcp_name)

        db.add(hcp)

        db.commit()

        db.refresh(hcp)

    interaction = Interaction(

        hcp_id=hcp.id,

        interaction_type=interaction_type,

        product=product,

        notes=notes,

        date=date,

        follow_up=follow_up

    )

    db.add(interaction)

    db.commit()

    return "Interaction logged successfully"



@tool
def edit_interaction(interaction_id: int, field: str, value: str):

    db = SessionLocal()

    interaction = db.query(Interaction).filter(
        Interaction.id == interaction_id
    ).first()

    if not interaction:

        return "Interaction not found"

    setattr(interaction, field, value)

    db.commit()

    return "Interaction updated"



@tool
def interaction_history(hcp_name: str):

    db = SessionLocal()

    hcp = db.query(HCP).filter(HCP.name == hcp_name).first()

    if not hcp:

        return "No interactions found"

    interactions = db.query(Interaction).filter(
        Interaction.hcp_id == hcp.id
    ).all()

    result = []

    for i in interactions:

        result.append(
            f"{i.date}: {i.product} - {i.notes}"
        )

    return result



@tool
def suggest_followup(notes: str):

    if "research" in notes.lower():

        return "Send clinical trial documents"

    if "sample" in notes.lower():

        return "Provide drug samples"

    return "Schedule next visit"