# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import argparse
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from rich.console import Console

from llama_models.cli.verify_download import (
    calculate_md5,
    load_checksums,
    run_verify_cmd,
    verify_files,
)


class CalculateMd5Tests(unittest.TestCase):
    def test_md5_known_vector(self):
        content = b"hello world\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "file.bin"
            filepath.write_bytes(content)
            self.assertEqual(
                calculate_md5(filepath),
                hashlib.md5(content).hexdigest(),
            )


class LoadChecksumsTests(unittest.TestCase):
    def test_md5_format(self):
        content = (
            "8f0bfb9428f98a4d52c294aadb485f66  consolidated.00.pth\n"
            "618025eb0e972cabc8f5aae54c91b855  params.json\n"
            "08292403f8b173e7524d7fba7bbbd2d3  tokenizer.model\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checklist_path = Path(tmpdir) / "checklist.chk"
            checklist_path.write_text(content)
            checksums = load_checksums(checklist_path)
        self.assertEqual(checksums["consolidated.00.pth"], "8f0bfb9428f98a4d52c294aadb485f66")
        self.assertEqual(checksums["params.json"], "618025eb0e972cabc8f5aae54c91b855")
        self.assertEqual(checksums["tokenizer.model"], "08292403f8b173e7524d7fba7bbbd2d3")

    def test_leading_dot_slash_prefix(self):
        content = "8f0bfb9428f98a4d52c294aadb485f66  ./consolidated.00.pth\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            checklist_path = Path(tmpdir) / "checklist.chk"
            checklist_path.write_text(content)
            checksums = load_checksums(checklist_path)
        self.assertEqual(list(checksums.keys()), ["consolidated.00.pth"])

    def test_blank_lines_ignored(self):
        content = "\n8f0bfb9428f98a4d52c294aadb485f66  params.json\n\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            checklist_path = Path(tmpdir) / "checklist.chk"
            checklist_path.write_text(content)
            checksums = load_checksums(checklist_path)
        self.assertEqual(list(checksums.keys()), ["params.json"])


class VerifyFilesTests(unittest.TestCase):
    def _checksums(self, model_dir: Path, content: bytes) -> dict[str, str]:
        (model_dir / "consolidated.00.pth").write_bytes(content)
        return {"consolidated.00.pth": hashlib.md5(content).hexdigest()}

    def test_match(self):
        content = b"model weights"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            checksums = self._checksums(model_dir, content)
            results = verify_files(model_dir, checksums, Console())
        self.assertTrue(results[0].matches)
        self.assertTrue(results[0].exists)

    def test_mismatch(self):
        content = b"model weights"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "consolidated.00.pth").write_bytes(content)
            checksums = {"consolidated.00.pth": hashlib.md5(b"different").hexdigest()}
            results = verify_files(model_dir, checksums, Console())
        self.assertFalse(results[0].matches)
        self.assertTrue(results[0].exists)

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            checksums = {"consolidated.00.pth": hashlib.md5(b"x").hexdigest()}
            results = verify_files(model_dir, checksums, Console())
        self.assertFalse(results[0].exists)
        self.assertFalse(results[0].matches)
        self.assertIsNone(results[0].actual_hash)


class RunVerifyCmdTests(unittest.TestCase):
    def _run_verify(self, model_dir: Path, content: bytes):
        (model_dir / "consolidated.00.pth").write_bytes(content)
        (model_dir / "checklist.chk").write_text(f"{hashlib.md5(content).hexdigest()}  consolidated.00.pth\n")
        args = argparse.Namespace(model_id="test-model")
        parser = argparse.ArgumentParser()
        with mock.patch(
            "llama_models.utils.model_utils.model_local_dir",
            return_value=str(model_dir),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_verify_cmd(args, parser)
        return stdout.getvalue()

    def test_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = self._run_verify(Path(tmpdir), b"model weights")
        self.assertIn("All files verified successfully", output)

    def test_mismatch(self):
        content = b"model weights"
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "consolidated.00.pth").write_bytes(content)
            (model_dir / "checklist.chk").write_text(f"{hashlib.md5(b'corrupt').hexdigest()}  consolidated.00.pth\n")
            args = argparse.Namespace(model_id="test-model")
            parser = argparse.ArgumentParser()
            with mock.patch(
                "llama_models.utils.model_utils.model_local_dir",
                return_value=str(model_dir),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    run_verify_cmd(args, parser)
            output = stdout.getvalue()
        self.assertIn("Hash mismatch", output)

    def test_missing_checklist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            args = argparse.Namespace(model_id="test-model")
            parser = argparse.ArgumentParser()
            with mock.patch(
                "llama_models.utils.model_utils.model_local_dir",
                return_value=str(model_dir),
            ):
                with self.assertRaises(SystemExit):
                    run_verify_cmd(args, parser)


if __name__ == "__main__":
    unittest.main()
