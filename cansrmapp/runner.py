#! /usr/bin/env python

import os
import time
import logging
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
from cansrmapp.exceptions import CansrmappError

import cansrmapp

logger = logging.getLogger(__name__)


class CansrmappRunner(object):
    """
    Class to run algorithm
    """
    def __init__(self, outdir=None,
                 exitcode=None,
                 skip_logging=True,
                 input_data_dict=None,
                 provenance_utils=ProvenanceUtil()):
        """
        Constructor

        :param outdir: Directory to create and put results in
        :type outdir: str
        :param skip_logging: If ``True`` skip logging, if ``None`` or ``False`` do NOT skip logging
        :type skip_logging: bool
        :param exitcode: value to return via :py:meth:`.CansrmappRunner.run` method
        :type int:
        :param input_data_dict: Command line arguments used to invoke this
        :type input_data_dict: dict
        :param provenance_utils: Wrapper for `fairscape-cli <https://pypi.org/project/fairscape-cli>`__
                                 which is used for
                                 `RO-Crate <https://www.researchobject.org/ro-crate>`__ creation and population
        :type provenance_utils: :py:class:`~cellmaps_utils.provenance.ProvenanceUtil`
        """
        if outdir is None:
            raise CansrmappError('outdir is None')

        self._outdir = os.path.abspath(outdir)
        self._exitcode = exitcode
        self._start_time = int(time.time())
        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging
        self._input_data_dict = input_data_dict
        self._provenance_utils = provenance_utils

        logger.debug('In constructor')

    def run(self):
        """
        Runs cansrmapp


        :return:
        """
        exitcode = 99
        try:
            logger.debug('In run method')
            if os.path.isdir(self._outdir):
                raise CansrmappError(self._outdir + ' already exists')
            if not os.path.isdir(self._outdir):
                os.makedirs(self._outdir, mode=0o755)
            if self._skip_logging is False:
                logutils.setup_filelogger(outdir=self._outdir,
                                          handlerprefix='cansrmapp')
            logutils.write_task_start_json(outdir=self._outdir,
                                           start_time=self._start_time,
                                           data={'commandlineargs': self._input_data_dict},
                                           version=cansrmapp.__version__)

            # set exit code to value passed in via constructor
            exitcode = self._exitcode
        finally:
            # write a task finish file
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)


        return exitcode
