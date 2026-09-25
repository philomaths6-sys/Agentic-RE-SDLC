# api/main.py
"""FastAPI entry point exposing requirement generation and SDLC recommendation.
Two endpoints:
- POST /requirement  -> returns a generated requirement for a user query.
- POST /sdlc        -> returns an SDLC plan for a given requirement.
Both use the same guardrails and LLM loading logic defined in the agents package.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import utility functions
from agents.requirement_generator import generate_requirement
from agents.sdlc_agent import recommend_sdlc

app = FastAPI(title="Agentic SDLC Service", version="0.1.0")

class RequirementRequest(BaseModel):
    query: str

class RequirementResponse(BaseModel):
    requirement: str

class SDLCRequest(BaseModel):
    requirement: str

class SDLCResponse(BaseModel):
    plan: str

@app.post("/requirement", response_model=RequirementResponse)
def create_requirement(req: RequirementRequest):
    try:
        req_text = generate_requirement(req.query)
        return RequirementResponse(requirement=req_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sdlc", response_model=SDLCResponse)
def create_sdlc_plan(req: SDLCRequest):
    try:
        plan = recommend_sdlc(req.requirement)
        return SDLCResponse(plan=plan)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
