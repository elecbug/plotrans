import contextlib
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper_localizer.cli import doctor,main


class CliTests(unittest.TestCase):
    def test_cli_has_no_summary_model_option(self):
        for command in ['run','doctor']:
            with self.subTest(command=command),contextlib.redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit) as exc:main([command,'--help'])
                self.assertEqual(exc.exception.code,0)
                self.assertNotIn('--summary-model',output.getvalue())
                self.assertNotIn('summarize',output.getvalue())

    def test_doctor_checks_only_the_translation_model(self):
        with patch('paper_localizer.cli.importlib.metadata.version',return_value='2.5.17'), \
             patch('paper_localizer.cli.shutil.which',return_value='/usr/bin/java'), \
             patch('paper_localizer.cli.subprocess.run',return_value=SimpleNamespace(returncode=0,stderr='Java 17')), \
             patch('paper_localizer.cli.OllamaClient') as client,contextlib.redirect_stdout(StringIO()):
            self.assertEqual(doctor('http://localhost:11434','translator'),0)
            client.return_value.check_models.assert_called_once_with('translator')
