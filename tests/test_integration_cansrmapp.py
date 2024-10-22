#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Integration Tests for `cansrmapp` package."""

import os

import unittest
from cansrmapp import cansrmappcmd

SKIP_REASON = 'CANSRMAPP_INTEGRATION_TEST ' \
              'environment variable not set, cannot run integration ' \
              'tests'

@unittest.skipUnless(os.getenv('CANSRMAPP_INTEGRATION_TEST') is not None, SKIP_REASON)
class TestIntegrationCansrmapp(unittest.TestCase):
    """Tests for `cansrmapp` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_something(self):
        """Tests parse arguments"""
        self.assertEqual(1, 1)
