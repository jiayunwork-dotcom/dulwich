# test_bisect.py -- Tests for bisect functionality
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

"""Tests for bisect functionality."""

import os
import shutil
import tempfile

from dulwich.bisect import (
    STEP_FIRST_BAD,
    STEP_NEXT,
    STEP_ONLY_SKIPPED,
    BisectState,
)
from dulwich.repo import Repo
from dulwich.tests.utils import make_commit

from . import TestCase


class BisectStateTests(TestCase):
    """Tests for BisectState class."""

    def setUp(self) -> None:
        super().setUp()
        self.test_dir = tempfile.mkdtemp()
        self.repo = Repo.init(self.test_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_is_active_false(self) -> None:
        """Test is_active when no bisect session is active."""
        state = BisectState(self.repo)
        self.assertFalse(state.is_active)

    def test_start_bisect(self) -> None:
        """Test starting a bisect session."""
        # Create at least one commit so HEAD exists
        c1 = make_commit(id=b"1" * 40, message=b"initial commit")
        self.repo.object_store.add_object(c1)
        self.repo.refs[b"HEAD"] = c1.id
        self.repo.refs[b"refs/heads/main"] = c1.id

        state = BisectState(self.repo)
        state.start()

        self.assertTrue(state.is_active)
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_START"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_TERMS"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_NAMES"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_LOG"))
        )

    def test_start_bisect_no_head(self) -> None:
        """Test starting a bisect session when repository has no HEAD."""
        state = BisectState(self.repo)

        with self.assertRaises(ValueError) as cm:
            state.start()
        self.assertIn("Cannot start bisect: repository has no HEAD", str(cm.exception))

    def test_start_bisect_already_active(self) -> None:
        """Test starting a bisect session when one is already active."""
        # Create at least one commit so HEAD exists
        c1 = make_commit(id=b"1" * 40, message=b"initial commit")
        self.repo.object_store.add_object(c1)
        self.repo.refs[b"HEAD"] = c1.id

        state = BisectState(self.repo)
        state.start()

        with self.assertRaises(ValueError):
            state.start()

    def test_mark_bad_no_session(self) -> None:
        """Test marking bad commit when no session is active."""
        state = BisectState(self.repo)

        with self.assertRaises(ValueError):
            state.mark_bad()

    def test_mark_good_no_session(self) -> None:
        """Test marking good commit when no session is active."""
        state = BisectState(self.repo)

        with self.assertRaises(ValueError):
            state.mark_good()

    def test_reset_no_session(self) -> None:
        """Test resetting when no session is active."""
        state = BisectState(self.repo)

        with self.assertRaises(ValueError):
            state.reset()

    def test_bisect_workflow(self) -> None:
        """Test a complete bisect workflow."""
        # Create some commits
        c1 = make_commit(message=b"good commit 1")
        c2 = make_commit(message=b"good commit 2", parents=[c1.id])
        c3 = make_commit(message=b"bad commit", parents=[c2.id])
        c4 = make_commit(message=b"bad commit 2", parents=[c3.id])

        # Add commits to object store
        for commit in [c1, c2, c3, c4]:
            self.repo.object_store.add_object(commit)

        # Set HEAD to latest commit
        self.repo.refs[b"HEAD"] = c4.id

        # Start bisect
        state = BisectState(self.repo)
        state.start()

        # Mark bad and good
        state.mark_bad(c4.id)
        state.mark_good(c1.id)

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
                    f"good-{c1.id.decode('ascii')}",
                )
            )
        )

        # Reset
        state.reset()
        self.assertFalse(state.is_active)

    def _make_linear_commits(self, n=6):
        """Create a linear chain of n commits and point HEAD at the tip."""
        commits = []
        parents = []
        for i in range(1, n + 1):
            commit = make_commit(message=f"c{i}".encode(), parents=parents)
            self.repo.object_store.add_object(commit)
            commits.append(commit)
            parents = [commit.id]
        self.repo.refs[b"HEAD"] = commits[-1].id
        self.repo.refs[b"refs/heads/main"] = commits[-1].id
        return commits

    def _bisect_head(self):
        with open(os.path.join(self.repo.controldir(), "BISECT_HEAD"), "rb") as f:
            return f.read().strip()

    def test_no_checkout_creates_bisect_head(self) -> None:
        commits = self._make_linear_commits()
        state = BisectState(self.repo)
        state.start(no_checkout=True)
        self.assertTrue(state.no_checkout)
        self.assertEqual(commits[-1].id, self._bisect_head())

    def test_no_checkout_does_not_move_head(self) -> None:
        commits = self._make_linear_commits()
        original_head = self.repo.head()
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id], no_checkout=True)
        # BISECT_HEAD points at the next commit, HEAD is untouched.
        self.assertTrue(state.no_checkout)
        self.assertEqual(original_head, self.repo.head())
        bisect_head = self._bisect_head()

        # Marking the selected commit bad uses BISECT_HEAD by default and
        # still does not move HEAD.
        next_commit = state.mark_bad()
        self.assertIsNotNone(next_commit)
        self.assertEqual(bisect_head, state.bad_commits()[0])
        self.assertEqual(original_head, self.repo.head())
        self.assertNotEqual(bisect_head, self._bisect_head())

        # Reset leaves HEAD in place but removes BISECT_HEAD.
        state.reset()
        self.assertEqual(original_head, self.repo.head())
        self.assertFalse(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_HEAD"))
        )

    def test_no_checkout_log_marks_option(self) -> None:
        self._make_linear_commits()
        state = BisectState(self.repo)
        state.start(no_checkout=True)
        self.assertIn("git bisect start '--no-checkout'", state.get_log())

    def test_normal_session_log_unchanged(self) -> None:
        self._make_linear_commits()
        state = BisectState(self.repo)
        state.start()
        log = state.get_log()
        self.assertIn("git bisect start\n", log)
        self.assertNotIn("--no-checkout", log)
        self.assertFalse(state.no_checkout)

    def test_skip_does_not_exclude_untested_commits(self) -> None:
        commits = self._make_linear_commits(6)
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id])
        midpoint = state.last_step.commit

        state.skip([midpoint])

        skipped = set(state.skip_commits())
        candidates = state._find_bisect_candidates(
            commits[-1].id, [commits[0].id], skipped
        )
        # Ancestors beyond the skipped commit remain candidates ...
        for commit in commits[1:-1]:
            self.assertIn(commit.id, candidates)
        # ... but skipped commits do not participate in the decision.
        self.assertNotIn(state.last_step.commit, skipped)
        self.assertEqual(STEP_NEXT, state.last_step.kind)

    def test_skip_everything_reports_only_skipped(self) -> None:
        commits = self._make_linear_commits(4)
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id])

        result = None
        for _ in range(10):
            result = state.last_step
            if result is None or result.commit is None:
                break
            state.skip([result.commit])

        self.assertEqual(STEP_ONLY_SKIPPED, state.last_step.kind)
        possible = state.last_step.possible
        self.assertIn(commits[-1].id, possible)
        self.assertTrue(set(possible) > {commits[-1].id})
        log = state.get_log()
        self.assertIn("# only skipped commits left to test", log)
        self.assertIn("# possible first bad commit:", log)
        state.reset()

    def test_first_bad_commit_found(self) -> None:
        commits = self._make_linear_commits(3)
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id])
        self.assertEqual(commits[1].id, state.last_step.commit)
        state.mark_bad(commits[1].id)
        self.assertEqual(STEP_FIRST_BAD, state.last_step.kind)
        self.assertEqual(commits[1].id, state.last_step.commit)
        self.assertIn("# first bad commit:", state.get_log())

    def test_replay_preserves_no_checkout(self) -> None:
        commits = self._make_linear_commits(4)
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id], no_checkout=True)
        state.mark_bad()
        log = state.get_log()
        state.reset()

        state.replay(log)
        self.assertTrue(state.no_checkout)
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_HEAD"))
        )

    def test_reset_removes_bisect_head(self) -> None:
        commits = self._make_linear_commits()
        state = BisectState(self.repo)
        state.start(bad=commits[-1].id, good=[commits[0].id], no_checkout=True)
        self.assertTrue(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_HEAD"))
        )
        state.reset()
        self.assertFalse(
            os.path.exists(os.path.join(self.repo.controldir(), "BISECT_HEAD"))
        )
