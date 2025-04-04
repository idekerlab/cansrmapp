#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""Tests for `cansrmapp` package."""
import os
import tempfile
import shutil

import unittest
import cansrmapp


class Testcansrmapp(unittest.TestCase):
    """Tests for `cansrmapp` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_summarize_random_states(self):
        """Tests constructor"""
        res = cansrmapp.summarize_random_states()
        self.assertTrue('NumPy: ' in res)


