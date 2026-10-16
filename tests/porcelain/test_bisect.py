# test_bisect.py -- tests for porcelain bisect
# Copyright (C) 2025 Jelmer Vernooij <jelmer@jelmer.uk>
#
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# Dulwich is dual-licensed under the Apache License, Version 2.0 and the GNU
# General Public License as published by the Free Software Foundation; version 2.0
# or (at your option) any later version. You can redistribute it and/or
# modify it under the terms of either of these two licenses.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# You should have received a copy of the licenses; if not, see
# <http://www.gnu.org/licenses/> for a copy of the GNU General Public License
# and <http://www.apache.org/licenses/LICENSE-2.0> for a copy of the Apache
# License, Version 2.0.
#

"""Tests for porcelain bisect functions."""

import os
import shutil
import tempfile

from dulwich import porcelain
from dulwich.objects import Tree
from dulwich.repo import Repo
from dulwich.tests.utils import make_commit

from .. import TestCase


class BisectPorcelainTests(TestCase):
    """Tests for porcelain bisect functions."""

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.repo = porcelain.init(self.test_dir)

        # Create tree objects
        tree = Tree()
        self.repo.object_store.add_object(tree)

        # Create some commits with proper trees
        self.c1 = make_commit(message=b"initial commit", tree=tree.id)
        self.c2 = make_commit(
            message=b"second commit", parents=[self.c1.id], tree=tree.id
        )
        self.c3 = make_commit(
            message=b"third commit", parents=[self.c2.id], tree=tree.id
        )
        self.c4 = make_commit(
            message=b"fourth commit", parents=[self.c3.id], tree=tree.id
        )

        # Add commits to object store
        for commit in [self.c1, self.c2, self.c3, self.c4]:
            self.repo.object_store.add_object(commit)

        # Set HEAD to latest commit
        self.repo.refs[b"HEAD"] = self.c4.id
        self.repo.refs[b"refs/heads/master"] = self.c4.id

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_bisect_start(self) -> None:
        """Test bisect_start porcelain function."""
        porcelain.bisect_start(self.test_dir)

        # Check that bisect state files exist
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_START"))
        )

    def test_bisect_bad_good(self) -> None:
        """Test marking commits as bad and good."""
        porcelain.bisect_start(self.test_dir)
        porcelain.bisect_bad(self.test_dir, self.c4.id.decode("ascii"))
        porcelain.bisect_good(self.test_dir, self.c1.id.decode("ascii"))

        # Check that refs were created
        self.assertTrue(
            os.path.exists(
                os.path.join(self.repo.controldir(), "refs", "bisect", "bad")
            )
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    self.repo.controldir(),
                    "refs",
                    "bisect",
                    f"good-{self.c1.id.decode('ascii')}",
                )
            )
        )

    def test_bisect_log(self) -> None:
        """Test getting bisect log."""
        porcelain.bisect_start(self.test_dir)
        porcelain.bisect_bad(self.test_dir, self.c4.id.decode("ascii"))
        porcelain.bisect_good(self.test_dir, self.c1.id.decode("ascii"))

        log = porcelain.bisect_log(self.test_dir)

        self.assertIn("git bisect start", log)
        self.assertIn("git bisect bad", log)
        self.assertIn("git bisect good", log)

    def test_bisect_reset(self) -> None:
        """Test resetting bisect state."""
        porcelain.bisect_start(self.test_dir)
        porcelain.bisect_bad(self.test_dir)
        porcelain.bisect_good(self.test_dir, self.c1.id.decode("ascii"))

        porcelain.bisect_reset(self.test_dir)

        # Check that bisect state files are removed
        self.assertFalse(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_START"))
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.repo.controldir(), "refs", "bisect"))
        )


class BisectNoCheckoutTests(TestCase):
    """Tests for session-wide --no-checkout behavior."""

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.repo = porcelain.init(self.test_dir)
        self.commits = []
        for i in range(1, 7):
            with open(os.path.join(self.test_dir, "f"), "w") as f:
                f.write(f"v{i}\n")
            porcelain.add(self.test_dir, paths=[b"f"])
            self.commits.append(
                porcelain.commit(
                    self.test_dir,
                    message=f"c{i}",
                    author=b"Test <test@example.com>",
                    committer=b"Test <test@example.com>",
                )
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_no_checkout_uses_bisect_head(self) -> None:
        c1, _c2, _c3, _c4, _c5, c6 = self.commits
        original_head = self.repo.head()

        next_sha = porcelain.bisect_start(
            self.test_dir, bad=c6, good=c1, no_checkout=True
        )
        bisect_head_path = os.path.join(self.repo.controldir(), "BISECT_HEAD")
        self.assertTrue(os.path.exists(bisect_head_path))
        self.assertEqual(next_sha, open(bisect_head_path, "rb").read().strip())
        # HEAD and the working tree are untouched.
        self.assertEqual(original_head, self.repo.head())
        self.assertEqual("v6\n", open(os.path.join(self.test_dir, "f")).read())

        # Marking commits defaults to BISECT_HEAD and still doesn't
        # check anything out.
        porcelain.bisect_bad(self.test_dir)
        self.assertEqual(original_head, self.repo.head())
        self.assertEqual("v6\n", open(os.path.join(self.test_dir, "f")).read())
        porcelain.bisect_good(self.test_dir, c1)
        self.assertEqual(original_head, self.repo.head())
        self.assertEqual("v6\n", open(os.path.join(self.test_dir, "f")).read())

        porcelain.bisect_reset(self.test_dir)
        self.assertEqual(original_head, self.repo.head())
        self.assertFalse(os.path.exists(bisect_head_path))

    def test_no_checkout_survives_dirty_worktree(self) -> None:
        """Uncommitted/untracked files are safe throughout a no-checkout run."""
        c1, _c2, _c3, _c4, _c5, c6 = self.commits
        with open(os.path.join(self.test_dir, "f"), "w") as f:
            f.write("uncommitted\n")
        with open(os.path.join(self.test_dir, "untracked"), "w") as f:
            f.write("untracked\n")

        porcelain.bisect_start(self.test_dir, bad=c6, good=c1, no_checkout=True)
        porcelain.bisect_bad(self.test_dir)
        porcelain.bisect_good(self.test_dir, c1)
        porcelain.bisect_skip(self.test_dir)
        self.assertEqual("uncommitted\n", open(os.path.join(self.test_dir, "f")).read())
        self.assertEqual(
            "untracked\n",
            open(os.path.join(self.test_dir, "untracked")).read(),
        )

    def test_bare_repo_defaults_to_no_checkout(self) -> None:
        c1, _c2, _c3, _c4, _c5, c6 = self.commits
        bare_path = self.test_dir + ".git"
        porcelain.clone(self.test_dir, bare_path, bare=True)
        self.addCleanup(shutil.rmtree, bare_path)
        bare = Repo(bare_path)
        head = bare.head()

        next_sha = porcelain.bisect_start(bare_path, bad=c6, good=c1)
        self.assertTrue(os.path.exists(os.path.join(bare_path, "BISECT_HEAD")))
        self.assertEqual(
            next_sha, open(bare_path + "/BISECT_HEAD", "rb").read().strip()
        )
        self.assertEqual(head, bare.head())
        porcelain.bisect_reset(bare_path)


class BisectRunTests(TestCase):
    """Tests for ``porcelain.bisect_run``."""

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.repo = porcelain.init(self.test_dir)
        self.commits = []
        for i in range(1, 8):
            with open(os.path.join(self.test_dir, "f"), "w") as f:
                f.write(f"v{i}\n")
            porcelain.add(self.test_dir, paths=[b"f"])
            self.commits.append(
                porcelain.commit(
                    self.test_dir,
                    message=f"c{i}",
                    author=b"Test <test@example.com>",
                    committer=b"Test <test@example.com>",
                )
            )
        self.script_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.script_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def _script(self, content: str, name: str = "test.sh") -> str:
        path = os.path.join(self.script_dir, name)
        with open(path, "w") as f:
            f.write("#!/bin/sh\n" + content + "\n")
        os.chmod(path, 0o755)
        return path

    def _bad_ref(self) -> bytes:
        with open(
            os.path.join(self.repo.controldir(), "refs", "bisect", "bad"), "rb"
        ) as f:
            return f.read().strip()

    def test_run_finds_first_bad_commit(self) -> None:
        c1, _c2, _c3, c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        # Good iff the checked-out tree is at or before c3.
        script = self._script("test \"$(cat f | tr -d 'v\\n')\" -le 3")
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(0, rc)
        self.assertEqual(c4, self._bad_ref())
        # The session is left in place until an explicit reset.
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_START"))
        )
        porcelain.bisect_reset(self.test_dir)
        self.assertFalse(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_START"))
        )

    def test_run_aborts_on_128_plus_and_is_resumable(self) -> None:
        c1, _c2, _c3, c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        expected = open(
            os.path.join(self.repo.controldir(), "BISECT_EXPECTED_REV"), "rb"
        ).read()
        script = self._script("exit 130")
        rc = porcelain.bisect_run(self.test_dir, [script])
        # git maps the negated internal code through the exit status.
        self.assertEqual((-130) & 0xFF, rc)
        # No commits were marked and the session can be resumed.
        refs = os.listdir(os.path.join(self.repo.controldir(), "refs", "bisect"))
        self.assertEqual({"bad", f"good-{c1.decode('ascii')}"}, set(refs))
        self.assertEqual(
            expected,
            open(
                os.path.join(self.repo.controldir(), "BISECT_EXPECTED_REV"), "rb"
            ).read(),
        )

        script = self._script("test \"$(cat f | tr -d 'v\\n')\" -le 3")
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(0, rc)
        self.assertEqual(c4, self._bad_ref())
        porcelain.bisect_reset(self.test_dir)

    def test_run_exit_125_skips(self) -> None:
        c1, _c2, _c3, _c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        script = self._script("exit 125")
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(2, rc)
        refs = os.listdir(os.path.join(self.repo.controldir(), "refs", "bisect"))
        self.assertTrue(any(r.startswith("skip-") for r in refs))
        log = porcelain.bisect_log(self.test_dir)
        self.assertIn("# only skipped commits left to test", log)
        porcelain.bisect_reset(self.test_dir)

    def test_run_exit_126_bogus_on_good_revision(self) -> None:
        c1, _c2, _c3, _c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        midpoint = self.repo.head()
        script = self._script("exit 126")
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(1, rc)
        refs = os.listdir(os.path.join(self.repo.controldir(), "refs", "bisect"))
        self.assertEqual({"bad", f"good-{c1.decode('ascii')}"}, set(refs))
        # The tested revision is checked back out after verification.
        self.assertEqual(midpoint, self.repo.head())
        porcelain.bisect_reset(self.test_dir)

    def test_run_exit_126_when_script_works_on_good(self) -> None:
        c1, _c2, _c3, c4, _c5, _c6, c7 = self.commits
        marker = os.path.join(self.script_dir, "ran")
        if os.path.exists(marker):
            os.remove(marker)
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        script = self._script(
            "if [ ! -e '"
            + marker
            + "' ]; then touch '"
            + marker
            + "'; exit 126; fi; exit 0"
        )
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(0, rc)
        self.assertEqual(c4, self._bad_ref())
        os.remove(marker)
        porcelain.bisect_reset(self.test_dir)

    def test_run_requires_active_session_with_endpoints(self) -> None:
        script = self._script("exit 0")
        with self.assertRaises(porcelain.Error):
            porcelain.bisect_run(self.test_dir, [script])

        porcelain.bisect_start(self.test_dir)
        with self.assertRaises(porcelain.Error):
            porcelain.bisect_run(self.test_dir, [script])
        porcelain.bisect_reset(self.test_dir)

    def test_run_requires_command(self) -> None:
        c1, _c2, _c3, _c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        with self.assertRaises(porcelain.Error):
            porcelain.bisect_run(self.test_dir, [])
        porcelain.bisect_reset(self.test_dir)

    def test_run_no_checkout_via_bisect_head(self) -> None:
        c1, _c2, _c3, c4, _c5, _c6, c7 = self.commits
        good_shas = ",".join(s.decode("ascii") for s in (c1, *self.commits[1:3]))
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1, no_checkout=True)
        original_head = self.repo.head()
        controldir = self.repo.controldir()
        script = self._script(
            "case ',"
            + good_shas
            + ",' in *\"$(cat '"
            + os.path.join(controldir, "BISECT_HEAD")
            + "')\"*) exit 0;; *) exit 1;; esac"
        )
        rc = porcelain.bisect_run(self.test_dir, [script])
        self.assertEqual(0, rc)
        self.assertEqual(c4, self._bad_ref())
        self.assertEqual(original_head, self.repo.head())
        self.assertEqual("v7\n", open(os.path.join(self.test_dir, "f")).read())
        porcelain.bisect_reset(self.test_dir)

    def test_run_shell_command_string(self) -> None:
        c1, _c2, _c3, c4, _c5, _c6, c7 = self.commits
        porcelain.bisect_start(self.test_dir, bad=c7, good=c1)
        rc = porcelain.bisect_run(
            self.test_dir,
            "test \"$(cat f | tr -d 'v\\n')\" -le 3",
        )
        self.assertEqual(0, rc)
        self.assertEqual(c4, self._bad_ref())
        porcelain.bisect_reset(self.test_dir)
