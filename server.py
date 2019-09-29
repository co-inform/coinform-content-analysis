from fastapi import FastAPI
from routers import rumour_verif

app = FastAPI()

app.include_router(rumour_verif.router)


@app.get('/')
async def root():
    return {
        'name': 'coinform-content-analysis',
        'docs': '/docs'
    }
