from fastapi import FastAPI

app = FastAPI()

@app.get("/user")
def read_root():
    return {"message": "Hello World"}
