# bisect.py -- Git bisect algorithm implementation
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

"""Git bisect implementation."""

__all__ = [
    "STEP_FIRST_BAD",
    "STEP_NEXT",
    "STEP_ONLY_SKIPPED",
    "STEP_WAITING",
    "BisectState",
    "BisectStep",
]

import os
import shlex
from collections.abc import Sequence, Set
from dataclasses import dataclass, field

from dulwich.object_store import peel_sha
from dulwich.objects import Commit, ObjectID
from dulwich.refs import HEADREF, Ref
from dulwich.repo import Repo

#: Session does not yet have both a bad and a good revision.
STEP_WAITING = "waiting"
#: Another commit has been selected for testing.
STEP_NEXT = "next"
#: The first bad commit has been determined.
STEP_FIRST_BAD = "first-bad"
#: Every commit left in the bisection range was skipped.
STEP_ONLY_SKIPPED = "only-skipped"


@dataclass
class BisectStep:
    """Result of advancing a bisection.

    Attributes:
        kind: One of the ``STEP_*`` constants
        commit: The next commit to test, or ``None`` when the bisection
            cannot advance
        possible: When ``kind`` is :data:`STEP_ONLY_SKIPPED`, the commits
            (including the known bad one) that could be the first bad
            commit
        count: Number of revisions in the current bisection range,
            including skipped revisions but excluding the bad revision and
            the good revisions
    """

    kind: str
    commit: ObjectID | None = None
    possible: tuple[ObjectID, ...] = field(default_factory=tuple)
    count: int = 0


class BisectState:
    """Manages the state of a bisect session."""

    def __init__(self, repo: Repo) -> None:
        """Initialize BisectState.

        Args:
            repo: Repository to perform bisect on
        """
        self.repo = repo
        self._bisect_dir = os.path.join(repo.controldir(), "BISECT_START")
        self._bisect_head_file = os.path.join(repo.controldir(), "BISECT_HEAD")
        self._last_step: BisectStep | None = None

    @property
    def is_active(self) -> bool:
        """Check if a bisect session is active."""
        return os.path.exists(self._bisect_dir)

    @property
    def no_checkout(self) -> bool:
        """Check whether this session avoids touching HEAD and the work tree.

        A session is in no-checkout mode while the ``BISECT_HEAD`` pseudo-ref
        exists, mirroring the way git persists ``--no-checkout`` (and the
        default used for bare repositories).
        """
        return os.path.exists(self._bisect_head_file)

    @property
    def last_step(self) -> BisectStep | None:
        """The result of the most recent bisection step, if any."""
        return self._last_step

    def start(
        self,
        bad: ObjectID | None = None,
        good: Sequence[ObjectID] | None = None,
        paths: Sequence[bytes] | None = None,
        no_checkout: bool = False,
        term_bad: str = "bad",
        term_good: str = "good",
    ) -> None:
        """Start a new bisect session.

        Args:
            bad: The bad commit SHA (defaults to HEAD)
            good: List of good commit SHAs
            paths: Optional paths to limit bisect to
            no_checkout: If True, don't checkout commits during bisect
            term_bad: Term to use for bad commits (default: "bad")
            term_good: Term to use for good commits (default: "good")
        """
        if self.is_active:
            raise ValueError("Bisect session already in progress")

        # Create bisect state directory
        bisect_refs_dir = os.path.join(self.repo.controldir(), "refs", "bisect")
        os.makedirs(bisect_refs_dir, exist_ok=True)

        # Store current branch/commit
        try:
            ref_chain, sha = self.repo.refs.follow(HEADREF)
            if sha is None:
                # No HEAD exists
                raise ValueError("Cannot start bisect: repository has no HEAD")
            # Use the first non-HEAD ref in the chain, or the SHA itself
            current_branch: Ref | ObjectID
            if len(ref_chain) > 1:
                current_branch = ref_chain[1]  # The actual branch ref
            else:
                current_branch = sha  # Detached HEAD
        except KeyError:
            # Detached HEAD
            try:
                current_branch = self.repo.head()
            except KeyError:
                # No HEAD exists - can't start bisect
                raise ValueError("Cannot start bisect: repository has no HEAD")

        # Write BISECT_START
        with open(self._bisect_dir, "wb") as f:
            f.write(current_branch)

        if no_checkout:
            # BISECT_HEAD marks a no-checkout session; it always points at
            # the next commit to test and is read instead of HEAD when no
            # revision is given to good/bad/skip.
            if current_branch.startswith(b"refs/"):
                start_sha = self.repo.refs[Ref(current_branch)]
            else:
                start_sha = ObjectID(current_branch)
            self._set_bisect_head(start_sha)

        # Write BISECT_TERMS
        terms_file = os.path.join(self.repo.controldir(), "BISECT_TERMS")
        with open(terms_file, "w") as f:
            f.write(f"{term_bad}\n{term_good}\n")

        # Write BISECT_NAMES (paths)
        names_file = os.path.join(self.repo.controldir(), "BISECT_NAMES")
        with open(names_file, "w") as f:
            if paths:
                f.write(
                    "\n".join(path.decode("utf-8", "replace") for path in paths) + "\n"
                )
            else:
                f.write("\n")

        # Initialize BISECT_LOG
        log_file = os.path.join(self.repo.controldir(), "BISECT_LOG")
        with open(log_file, "w") as f:
            if no_checkout:
                f.write("git bisect start '--no-checkout'\n")
            else:
                f.write("git bisect start\n")
            f.write("# status: waiting for both good and bad commits\n")

        # Mark bad commit if provided
        if bad is not None:
            self.mark_bad(bad)

        # Mark good commits if provided
        if good:
            for g in good:
                self.mark_good(g)

    def mark_bad(self, rev: ObjectID | None = None) -> ObjectID | None:
        """Mark a commit as bad.

        Args:
            rev: Commit SHA to mark as bad (defaults to HEAD, or to
                ``BISECT_HEAD`` in a no-checkout session)

        Returns:
            The SHA of the next commit to test, or None if bisect is complete
        """
        if not self.is_active:
            raise ValueError("No bisect session in progress")

        if rev is None:
            rev = self._current_rev()
        else:
            rev = peel_sha(self.repo.object_store, rev)[1].id

        # Write bad ref
        bad_ref_path = os.path.join(self.repo.controldir(), "refs", "bisect", "bad")
        with open(bad_ref_path, "wb") as f:
            f.write(rev + b"\n")

        # Update log
        self._append_to_log(
            f"# bad: [{rev.decode('ascii')}] {self._get_commit_subject(rev)}"
        )
        self._append_to_log(f"git bisect bad {rev.decode('ascii')}")

        return self._find_next_commit()

    def mark_good(self, rev: ObjectID | None = None) -> ObjectID | None:
        """Mark a commit as good.

        Args:
            rev: Commit SHA to mark as good (defaults to HEAD, or to
                ``BISECT_HEAD`` in a no-checkout session)

        Returns:
            The SHA of the next commit to test, or None if bisect is complete
        """
        if not self.is_active:
            raise ValueError("No bisect session in progress")

        if rev is None:
            rev = self._current_rev()
        else:
            rev = peel_sha(self.repo.object_store, rev)[1].id

        # Write good ref
        good_ref_path = os.path.join(
            self.repo.controldir(), "refs", "bisect", f"good-{rev.decode('ascii')}"
        )
        with open(good_ref_path, "wb") as f:
            f.write(rev + b"\n")

        # Update log
        self._append_to_log(
            f"# good: [{rev.decode('ascii')}] {self._get_commit_subject(rev)}"
        )
        self._append_to_log(f"git bisect good {rev.decode('ascii')}")

        return self._find_next_commit()

    def skip(self, revs: Sequence[ObjectID] | None = None) -> ObjectID | None:
        """Skip one or more commits.

        Args:
            revs: List of commits to skip (defaults to [HEAD], or to
                ``BISECT_HEAD`` in a no-checkout session)

        Returns:
            The SHA of the next commit to test, or None if bisect is complete
        """
        if not self.is_active:
            raise ValueError("No bisect session in progress")

        if revs is None:
            revs = [self._current_rev()]

        for rev in revs:
            rev = peel_sha(self.repo.object_store, rev)[1].id
            skip_ref_path = os.path.join(
                self.repo.controldir(), "refs", "bisect", f"skip-{rev.decode('ascii')}"
            )
            with open(skip_ref_path, "wb") as f:
                f.write(rev + b"\n")

            self._append_to_log(f"git bisect skip {rev.decode('ascii')}")

        return self._find_next_commit()

    def reset(self, commit: ObjectID | None = None) -> None:
        """Reset bisect state and return to original branch/commit.

        In a no-checkout session HEAD and the working tree are left
        untouched.

        Args:
            commit: Optional commit to reset to (defaults to original branch/commit)
        """
        if not self.is_active:
            raise ValueError("No bisect session in progress")

        no_checkout = self.no_checkout

        # Read original branch/commit
        with open(self._bisect_dir, "rb") as f:
            original = f.read().strip()

        # Clean up bisect files
        for filename in [
            "BISECT_START",
            "BISECT_TERMS",
            "BISECT_NAMES",
            "BISECT_LOG",
            "BISECT_EXPECTED_REV",
            "BISECT_ANCESTORS_OK",
            "BISECT_HEAD",
            "BISECT_RUN",
        ]:
            filepath = os.path.join(self.repo.controldir(), filename)
            if os.path.exists(filepath):
                os.remove(filepath)

        # Clean up refs/bisect directory
        bisect_refs_dir = os.path.join(self.repo.controldir(), "refs", "bisect")
        if os.path.exists(bisect_refs_dir):
            for filename in os.listdir(bisect_refs_dir):
                os.remove(os.path.join(bisect_refs_dir, filename))
            os.rmdir(bisect_refs_dir)

        if no_checkout:
            # No-checkout sessions never moved HEAD; leave it as-is, even
            # when an explicit commit was requested (matching git).
            return

        # Reset to target commit/branch
        if commit is None:
            if original.startswith(b"refs/"):
                # It's a branch reference - need to create a symbolic ref
                self.repo.refs.set_symbolic_ref(HEADREF, Ref(original))
            else:
                # It's a commit SHA
                self.repo.refs[HEADREF] = ObjectID(original)
        else:
            commit = peel_sha(self.repo.object_store, commit)[1].id
            self.repo.refs[HEADREF] = commit

    def get_log(self) -> str:
        """Get the bisect log."""
        if not self.is_active:
            raise ValueError("No bisect session in progress")

        log_file = os.path.join(self.repo.controldir(), "BISECT_LOG")
        with open(log_file) as f:
            return f.read()

    def replay(self, log_content: str) -> None:
        """Replay a bisect log.

        Args:
            log_content: The bisect log content to replay
        """
        # Parse and execute commands from log
        for line in log_content.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            # Log entries are shell-quoted, e.g.
            # "git bisect start '--no-checkout'"
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()
            if len(parts) < 3 or parts[0] != "git" or parts[1] != "bisect":
                continue

            cmd = parts[2]
            args = parts[3:] if len(parts) > 3 else []

            if cmd == "start":
                self.start(no_checkout="--no-checkout" in args)
            elif cmd == "bad":
                rev = ObjectID(args[0].encode("ascii")) if args else None
                self.mark_bad(rev)
            elif cmd == "good":
                rev = ObjectID(args[0].encode("ascii")) if args else None
                self.mark_good(rev)
            elif cmd == "skip":
                revs = [ObjectID(arg.encode("ascii")) for arg in args] if args else None
                self.skip(revs)

    def _current_rev(self) -> ObjectID:
        """The revision currently under test.

        Reads ``BISECT_HEAD`` in a no-checkout session, otherwise HEAD.
        """
        if self.no_checkout:
            with open(self._bisect_head_file, "rb") as f:
                return ObjectID(f.read().strip())
        return self.repo.head()

    def _set_bisect_head(self, rev: ObjectID) -> None:
        """Point BISECT_HEAD at the given revision."""
        with open(self._bisect_head_file, "wb") as f:
            f.write(rev + b"\n")

    @property
    def bisect_head(self) -> ObjectID | None:
        """The SHA recorded in ``BISECT_HEAD``.

        None when not bisecting in no-checkout mode.
        """
        if not self.no_checkout:
            return None
        with open(self._bisect_head_file, "rb") as f:
            return ObjectID(f.read().strip())

    def _refs(self, prefix: str) -> list[ObjectID]:
        """Read all refs/bisect/<prefix>-<sha> markers (or 'bad')."""
        bisect_refs_dir = os.path.join(self.repo.controldir(), "refs", "bisect")
        results: list[ObjectID] = []
        if not os.path.isdir(bisect_refs_dir):
            return results
        for filename in sorted(os.listdir(bisect_refs_dir)):
            if filename == prefix or filename.startswith(prefix + "-"):
                with open(os.path.join(bisect_refs_dir, filename), "rb") as f:
                    results.append(ObjectID(f.read().strip()))
        return results

    def bad_commits(self) -> list[ObjectID]:
        """Return the marked bad revision (at most one), if any."""
        return self._refs("bad")

    def good_commits(self) -> list[ObjectID]:
        """Return all marked good revisions."""
        return self._refs("good")

    def skip_commits(self) -> list[ObjectID]:
        """Return all skipped revisions."""
        return self._refs("skip")

    def _compute_next(self) -> BisectStep:
        """Compute the next bisection step without recording it."""
        bad_shas = self._refs("bad")
        if not bad_shas:
            return BisectStep(STEP_WAITING)
        bad_sha = bad_shas[0]

        good_shas = self._refs("good")
        if not good_shas:
            return BisectStep(STEP_WAITING)

        skip_shas = set(self._refs("skip"))

        # Find commits between good and bad (skipped commits are walked
        # through so that they never exclude untested ancestors)
        candidates = self._find_bisect_candidates(bad_sha, good_shas, skip_shas)

        if not candidates:
            # Nothing left between the endpoints: the bad revision itself
            # is the first bad commit.
            return BisectStep(
                STEP_FIRST_BAD, commit=bad_sha, possible=(bad_sha,), count=0
            )

        testable = [sha for sha in candidates if sha not in skip_shas]

        if not testable:
            # Every remaining revision was skipped.
            possible = (bad_sha, *[sha for sha in candidates if sha in skip_shas])
            return BisectStep(
                STEP_ONLY_SKIPPED, possible=possible, count=len(candidates)
            )

        # Find midpoint, avoiding skipped revisions: pick the testable
        # revision closest to the regular midpoint.
        mid_idx = len(candidates) // 2

        def distance(index: int) -> tuple[int, int]:
            return (abs(index - mid_idx), index)

        best_idx = min(
            (i for i, sha in enumerate(candidates) if sha not in skip_shas),
            key=distance,
        )
        next_commit = candidates[best_idx]
        return BisectStep(STEP_NEXT, commit=next_commit, count=len(candidates))

    def _find_next_commit(self) -> ObjectID | None:
        """Find the next commit to test using binary search.

        Returns:
            The SHA of the next commit to test, or None if bisect is complete
        """
        step = self._compute_next()
        self._last_step = step

        if step.kind == STEP_WAITING:
            if not self._refs("bad"):
                self._append_to_log("# status: waiting for both good and bad commits")
            else:
                self._append_to_log(
                    "# status: waiting for good commit(s), bad commit known"
                )
            return None

        if step.kind == STEP_FIRST_BAD:
            bad_sha = step.commit
            assert bad_sha is not None
            # Bisect complete - the first bad commit is found
            self._append_to_log(
                f"# first bad commit: [{bad_sha.decode('ascii')}] "
                f"{self._get_commit_subject(bad_sha)}"
            )
            return None

        if step.kind == STEP_ONLY_SKIPPED:
            self._append_to_log("# only skipped commits left to test")
            for sha in step.possible:
                self._append_to_log(
                    f"# possible first bad commit: [{sha.decode('ascii')}] "
                    f"{self._get_commit_subject(sha)}"
                )
            return None

        next_commit = step.commit
        assert next_commit is not None

        # Write BISECT_EXPECTED_REV
        expected_file = os.path.join(self.repo.controldir(), "BISECT_EXPECTED_REV")
        with open(expected_file, "wb") as f:
            f.write(next_commit + b"\n")

        if self.no_checkout:
            # Make the next revision available via BISECT_HEAD and leave
            # HEAD and the working tree alone.
            self._set_bisect_head(next_commit)

        # Update status in log
        steps_remaining = self._estimate_steps(step.count)
        self._append_to_log(
            f"Bisecting: {step.count - 1} revisions left to test after this "
            f"(roughly {steps_remaining} steps)"
        )
        self._append_to_log(
            f"[{next_commit.decode('ascii')}] {self._get_commit_subject(next_commit)}"
        )

        return next_commit

    def _find_bisect_candidates(
        self, bad_sha: ObjectID, good_shas: Sequence[ObjectID], skip_shas: Set[ObjectID]
    ) -> list[ObjectID]:
        """Find all commits between good and bad commits.

        Skipped commits remain part of the range (and are walked through),
        so they can never cause untested ancestors to be excluded.

        Args:
            bad_sha: The bad commit SHA
            good_shas: List of good commit SHAs
            skip_shas: Set of commits to skip

        Returns:
            List of candidate commit SHAs in traversal order, excluding the
            bad commit itself and all good commits
        """
        good = set(good_shas)
        candidates: list[ObjectID] = []
        visited: set[ObjectID] = set(good)
        queue: list[ObjectID] = [bad_sha]

        while queue:
            sha = queue.pop(0)
            if sha in visited:
                continue

            visited.add(sha)

            if sha != bad_sha:
                # The bad commit itself is not a test candidate.
                candidates.append(sha)

            commit = self.repo.object_store[sha]

            # Add parents to queue; skipped commits are traversed like any
            # other commit so their ancestors stay in the candidate set.
            if isinstance(commit, Commit):
                for parent in commit.parents:
                    if parent not in visited:
                        queue.append(parent)

        return candidates

    def _get_commit_subject(self, sha: ObjectID) -> str:
        """Get the subject line of a commit message."""
        obj = self.repo.object_store[sha]
        if isinstance(obj, Commit):
            message = obj.message.decode("utf-8", errors="replace")
            lines = message.split("\n")
            return lines[0] if lines else ""
        return ""

    def _append_to_log(self, line: str) -> None:
        """Append a line to the bisect log."""
        log_file = os.path.join(self.repo.controldir(), "BISECT_LOG")
        with open(log_file, "a") as f:
            f.write(line + "\n")

    def _estimate_steps(self, num_candidates: int) -> int:
        """Estimate the number of steps remaining in bisect."""
        if num_candidates <= 1:
            return 0
        steps = 0
        while num_candidates > 1:
            num_candidates //= 2
            steps += 1
        return steps
