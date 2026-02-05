from pydantic import BaseModel, Field

class Triple(BaseModel):
    head: str
    relation: str
    tail: str

class CleanerNodeOutputStructure(BaseModel):
    cleaned_result: str
    is_short: bool

class AgentNodeOutputStructure(BaseModel):
    keywords: list[str]
    triples: list[Triple]
    abstract: str
    title: str
    final_markdown: str