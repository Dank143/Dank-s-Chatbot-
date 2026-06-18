from pydantic import BaseModel


class CreateChatBody(BaseModel):
    title: str = "New Chat"
    model: str | None = None


class UpdateChatBody(BaseModel):
    title: str | None = None
    model: str | None = None
    starred: bool | None = None


class UpdateSettingsBody(BaseModel):
    provider: str = "nim"
    key: str | None = None
    base_url: str | None = None
    temperature: float | None = None


class DocumentAttachment(BaseModel):
    name: str
    text: str


class SendMessageBody(BaseModel):
    content: str = ""
    model: str | None = None
    images: list[str] | None = None
    documents: list[DocumentAttachment] | None = None
    web_search: bool = False
    client_time: str | None = None


class SaveAssistantBody(BaseModel):
    content: str


class RegenerateBody(BaseModel):
    model: str | None = None
    web_search: bool = False
    client_time: str | None = None


class VerifyKeyBody(BaseModel):
    provider: str = "nim"
    key: str
    base_url: str


class WarmupBody(BaseModel):
    model: str | None = None
