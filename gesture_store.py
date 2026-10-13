"""Keeps the hand shapes you have taught and recognises new ones.

Recognition is nearest-neighbours: the new hand shape is compared with every
saved sample, and the closest few vote for a word. There is no training step,
so a newly recorded word works immediately.
"""
import json
import os
import threading
import time

import numpy as np

NEIGHBOURS = 5
SAVE_EVERY_SECONDS = 3
FEATURE_COUNT = 63


class GestureStore:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        self._data = {}        # word -> list of 63-number samples
        self._matrix = None    # cached array of all samples
        self._labels = None    # word for each row of the matrix
        self._dirty = False
        self._last_save = 0.0
        self._load()

    # ---------- saving and loading ----------
    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            data = {}
            for label, samples in raw.items():
                good = [list(map(float, s)) for s in samples if len(s) == FEATURE_COUNT]
                if good:
                    data[str(label)] = good
            self._data = data
        except (OSError, ValueError, AttributeError, TypeError):
            # keep the unreadable file instead of overwriting it
            try:
                os.replace(self.path, self.path + ".bak")
            except OSError:
                pass
            self._data = {}

    def save(self):
        with self._lock:
            if not self._dirty:
                return
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
            self._dirty = False
            self._last_save = time.time()

    # ---------- changing the data ----------
    def add(self, label, features):
        if len(features) != FEATURE_COUNT:
            raise ValueError("wrong number of features")
        with self._lock:
            self._data.setdefault(label, []).append([round(float(v), 4) for v in features])
            self._matrix = None
            self._dirty = True
            if time.time() - self._last_save > SAVE_EVERY_SECONDS:
                self.save()
            return len(self._data[label])

    def delete(self, label):
        with self._lock:
            if label not in self._data:
                return False
            del self._data[label]
            self._matrix = None
            self._dirty = True
            self.save()
            return True

    def counts(self):
        with self._lock:
            return [{"label": k, "count": len(v)} for k, v in sorted(self._data.items())]

    # ---------- recognising ----------
    def predict(self, features):
        """Returns (word, confidence 0-1, distance to closest sample), or None if nothing is taught."""
        with self._lock:
            if self._matrix is None:
                labels, rows = [], []
                for label, samples in self._data.items():
                    labels += [label] * len(samples)
                    rows += samples
                if not rows:
                    return None
                self._matrix = np.asarray(rows, dtype=np.float32)
                self._labels = labels
            matrix, labels = self._matrix, self._labels

        distances = np.linalg.norm(matrix - np.asarray(features, dtype=np.float32), axis=1)
        k = min(NEIGHBOURS, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]

        votes = {}
        for i in nearest:
            votes[labels[i]] = votes.get(labels[i], 0.0) + 1.0 / (float(distances[i]) + 0.05)
        best = max(votes, key=votes.get)
        confidence = votes[best] / sum(votes.values())
        return best, confidence, float(distances[nearest].min())
