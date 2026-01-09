import asyncio
from typing import Any


class LatestTickQueue:
    def __init__(self) -> None:
        self._q: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)

    def offer(self, item: Any) -> None:
        if self._q.full():
            try:
                _ = self._q.get_nowait()
            except Exception:
                pass
        try:
            self._q.put_nowait(item)
        except Exception:
            pass

    def try_get_nowait(self) -> Any:
        try:
            return self._q.get_nowait()
        except Exception:
            return None

    async def get(self) -> Any:
        return await self._q.get()
