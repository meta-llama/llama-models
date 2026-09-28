import asyncio

import httpx
import pytest

from llama_models.cli.download import DownloadError, DownloadTask, ParallelDownloader


class InterruptedStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"cd"
        raise httpx.ReadError("connection dropped")


def run(coro):
    return asyncio.run(coro)


def test_retried_range_replaces_partial_bytes(tmp_path, monkeypatch):
    output = tmp_path / "model.pth"
    output.write_bytes(b"ab")
    task = DownloadTask("https://example.test/model.pth", str(output), total_size=6, downloaded_size=2)
    downloader = ParallelDownloader()
    task.task_id = downloader.progress.add_task("test", total=6)
    attempts = []

    async def no_wait(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    def handler(request):
        assert request.headers["Range"] == "bytes=2-5"
        assert request.headers["Accept-Encoding"] == "identity"
        attempts.append(request)
        headers = {"Content-Range": "bytes 2-5/6"}
        if len(attempts) == 1:
            return httpx.Response(206, headers=headers, stream=InterruptedStream())
        return httpx.Response(206, headers=headers, content=b"cdef")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await downloader.download_chunk(client, task, 2, 5)

    run(exercise())
    assert len(attempts) == 2
    assert output.read_bytes() == b"abcdef"
    assert task.downloaded_size == 6
    assert downloader.verify_file_integrity(task)


@pytest.mark.parametrize(
    "status,headers,content",
    [
        (200, {}, b"abcdef"),
        (206, {"Content-Range": "bytes 0-5/6"}, b"cdef"),
        (206, {"Content-Range": "bytes 2-5/6"}, b"cd"),
        (206, {"Content-Range": "bytes 2-5/6"}, b"cdefg"),
    ],
)
def test_rejects_invalid_or_incomplete_range(tmp_path, status, headers, content):
    output = tmp_path / "model.pth"
    output.write_bytes(b"ab")
    task = DownloadTask("https://example.test/model.pth", str(output), total_size=6, downloaded_size=2, max_retries=1)
    downloader = ParallelDownloader()
    task.task_id = downloader.progress.add_task("test", total=6)

    def handler(request):
        return httpx.Response(status, headers=headers, content=content)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await downloader.download_chunk(client, task, 2, 5)

    with pytest.raises(DownloadError, match="Failed to download chunk"):
        run(exercise())
    assert output.read_bytes().startswith(b"ab")
    assert len(output.read_bytes()) <= 6


def test_oversized_partial_download_restarts(tmp_path):
    output = tmp_path / "model.pth"
    output.write_bytes(b"old-corrupted-data")
    task = DownloadTask("https://example.test/model.pth", str(output), total_size=6)
    downloader = ParallelDownloader()
    run(downloader.prepare_download(task))
    assert task.downloaded_size == 0
    assert output.read_bytes() == b""
