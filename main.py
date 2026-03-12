from fastapi import FastAPI
from database import Base, engine
from routes import router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI CRM Backend"
)

app.include_router(router)


@app.get("/")

def root():

    return {"message": "AI CRM backend running"}