"""Explicit approval fixtures for write-path tests."""

from app.security.approvals import ApprovalRequest


class ApproveWrites:
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return request.tool == "file_write"
