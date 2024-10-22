#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cansrmapp` package."""
import os
import tempfile
import shutil

import unittest
from cansrmapp.runner import CansrmappRunner


class TestCansrmapprunner(unittest.TestCase):
    """Tests for `cansrmapp` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CansrmappRunner(outdir='foo', skip_logging=True,
                                                       exitcode=0)

        self.assertIsNotNone(myobj)

    def test_run(self):
        """ Tests run()"""
        temp_dir = tempfile.mkdtemp()
        try:
            myobj = CansrmappRunner(outdir=os.path.join(temp_dir, 'foo'),
                                                         skip_logging=True,
                                                         exitcode=4)
            self.assertEqual(4, myobj.run())
        finally:
            shutil.rmtree(temp_dir)
