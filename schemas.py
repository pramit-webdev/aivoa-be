from pydantic import BaseModel


class ChatRequest(BaseModel):

    message: str


class InteractionCreate(BaseModel):

    hcp_name: str

    interaction_type: str

    product: str

    notes: str

    date: str

    follow_up: str