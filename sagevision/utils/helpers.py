"""Small helper utilities used across modules."""
from typing import Iterable, Iterator, List, Any


def chunked(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield successive chunks of `size` from `iterable`."""
    buf: List[Any] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
