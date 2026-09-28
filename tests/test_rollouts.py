import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from core import rollouts


class NoiseControlsTest(unittest.TestCase):
    def test_cli_defaults_and_explicit_options(self):
        parser = argparse.ArgumentParser()
        rollouts.add_noise_arguments(parser)
        self.assertEqual(parser.parse_args([]).noise_std, 0.0)
        self.assertEqual(parser.parse_args(['--no-noise']).noise_std, 0.0)
        self.assertEqual(parser.parse_args(['--noise-std', '0.125']).noise_std, 0.125)
        for value in ('-1', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(['--noise-std', value])
        with self.assertRaises(SystemExit):
            parser.parse_args(['--noise-std', '0.1', '--no-noise'])

    def test_invalid_programmatic_noise(self):
        for mean, std in ((0, -1), (0, np.nan), (0, np.inf), (np.inf, 0)):
            with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                rollouts.run_samples(np.eye(2), np.eye(2), 2,
                                     noise_mean=mean, noise_std=std, workers=1)

    def test_noise_does_not_consume_walk_random_draws(self):
        def fake_walk(B, chop=None, noise=None, quantize_input=False):
            before = np.random.random()
            added = 0.0 if noise is None else noise(17).sum()
            after = np.random.random()
            return SimpleNamespace(Bz=np.array([before, after, added]),
                                   Bz_hat=np.array([before, after, added]))

        with patch.object(rollouts.gsw, 'gram_schmidt_walk', side_effect=fake_walk):
            _, _, clean, _ = rollouts.run_samples(np.eye(3), np.eye(3), 8, seed=7, workers=1)
            _, _, noisy, _ = rollouts.run_samples(np.eye(3), np.eye(3), 8,
                                                  noise_std=0.1, seed=7, workers=1)
            _, _, repeat, _ = rollouts.run_samples(np.eye(3), np.eye(3), 8,
                                                   noise_std=0.1, seed=7, workers=1)
        np.testing.assert_array_equal(clean[:2], noisy[:2])
        np.testing.assert_array_equal(noisy, repeat)
        self.assertTrue(np.any(noisy[2] != 0))
        np.testing.assert_array_equal(clean[2], 0)

    def test_serial_rollouts_preserve_callers_random_state(self):
        np.random.seed(123)
        expected = np.random.random(4)
        np.random.seed(123)
        rollouts.run_samples(np.eye(2), np.eye(2), 3, seed=5, workers=1)
        np.testing.assert_array_equal(np.random.random(4), expected)

    def test_worker_count_does_not_change_results(self):
        B = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.8]])
        for std in (0.0, 0.01):
            serial = rollouts.run_samples(B, np.eye(2), 6, seed=3, workers=1, noise_std=std)
            parallel = rollouts.run_samples(B, np.eye(2), 6, seed=3, workers=2, noise_std=std)
            for a, b in zip(serial, parallel):
                np.testing.assert_array_equal(a, b)


class CacheMetadataTest(unittest.TestCase):
    def test_noise_paths_are_distinct_from_legacy(self):
        clean = rollouts.output_path('figure.png', 0.0)
        noisy = rollouts.output_path('figure.png', 2**-32)
        self.assertNotEqual(clean, 'figure.png')
        self.assertNotEqual(clean, noisy)
        self.assertEqual(Path(clean).suffix, '.png')
        with self.assertRaises(ValueError):
            rollouts.output_path('figure.pdf', 0.0)

    def test_cache_checks_noise_and_configuration(self):
        metadata = rollouts.experiment_metadata(0.0, seed=3, num_samples=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npz'
            rollouts.save_cache(path, metadata, values=np.arange(3))
            cached = rollouts.load_cache(path, noise_std=0.0, seed=3, num_samples=10)
            np.testing.assert_array_equal(cached['values'], np.arange(3))
            self.assertEqual(json.loads(str(cached['metadata']))['noise_std'], 0.0)
            for expected in ({'noise_std': 0.1}, {'noise_mean': 0.1}, {'num_samples': 9}, {'seed': 4}):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    rollouts.load_cache(path, **expected)

    def test_legacy_cache_cannot_be_certified_by_requested_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.npz'
            np.savez(path, values=np.arange(3))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                rollouts.load_cache(path, noise_std=0.0)

    def test_metadata_records_provenance(self):
        metadata = rollouts.experiment_metadata(0.01, seed=4)
        self.assertEqual(metadata['noise_std'], 0.01)
        self.assertEqual(metadata['noise_mean'], 0.0)
        for key in ('git_revision', 'git_dirty', 'rng_scheme', 'created_utc'):
            self.assertIn(key, metadata)

    def test_figure_records_metadata(self):
        fig = Mock()
        metadata = rollouts.experiment_metadata(0.01, seed=4)
        rollouts.save_figure(fig, 'figure.png', metadata)
        saved = json.loads(fig.savefig.call_args.kwargs['metadata']['Description'])
        self.assertEqual(saved, metadata)


class SweepIntegrationTest(unittest.TestCase):
    @staticmethod
    def fake_samples(B, directions, num_samples, **options):
        bz = np.random.default_rng(1).normal(size=(B.shape[0], num_samples))
        bz *= 1 + options['noise_std']
        return np.abs(bz).mean(axis=0), directions @ bz, bz, bz

    def test_size_sweep_records_noise_and_replots_without_sampling(self):
        import n_subgauss

        with tempfile.TemporaryDirectory() as directory:
            for std in (0.0, 0.01):
                options = dict(matrix='clustered', n_values=[6], sig_bits_values=[3, 52],
                               num_samples=10, num_dirs=4, workers=1, seed=0,
                               save_path=str(Path(directory) / 'size.png'), noise_std=std)
                with patch.object(rollouts, 'run_samples', side_effect=self.fake_samples) as sample:
                    n_subgauss.n_subgauss(**options)
                    self.assertEqual(sample.call_count, 2)
                    self.assertEqual([call.kwargs['noise_std'] for call in sample.call_args_list], [std, std])
                    image = Path(rollouts.output_path(options['save_path'], std))
                    self.assertTrue(image.exists())
                    cache = image.with_name(image.stem + '_cache.npz')
                    before = rollouts.load_cache(cache, noise_std=std)
                    sample.reset_mock()
                    n_subgauss.n_subgauss(**options, plot_only=True)
                    sample.assert_not_called()
                    after = rollouts.load_cache(cache, noise_std=std)
                    self.assertEqual(rollouts.metadata_from_cache(before), rollouts.metadata_from_cache(after))

    def test_bit_sweep_records_noise_and_replots_without_sampling(self):
        import bits_subgauss

        B = np.tile(np.array([[0.6], [0.8]]), (1, 6))
        with tempfile.TemporaryDirectory() as directory:
            options = dict(n=6, sig_bits_values=[3, 12, 52], num_samples=10, num_dirs=4,
                           workers=1, seed=0, noise_std=0.02,
                           save_path=str(Path(directory) / 'bits.png'))
            with patch('core.matrices.matrix_family', return_value=lambda n: B), \
                    patch.object(rollouts, 'run_samples', side_effect=self.fake_samples) as sample:
                bits_subgauss.bits_subgauss(**options)
                self.assertEqual(sample.call_count, 3)
                self.assertTrue(all(call.kwargs['noise_std'] == 0.02 for call in sample.call_args_list))
                image = Path(rollouts.output_path(options['save_path'], 0.02))
                self.assertTrue(image.exists())
                sample.reset_mock()
                bits_subgauss.bits_subgauss(**options, plot_only=True)
                sample.assert_not_called()

    def test_noise_comparison_attributes_current_rounding_source(self):
        import noise_vs_chop

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'rounding.npz'
            output = Path(directory) / 'comparison.png'
            config = dict(seed=0, matrix='higgs', matrix_n=6, num_samples=10, num_dirs=4,
                          t=3.0, sig_bits=[3, 52], n_values=[6])

            def write_source(revision, value):
                metadata = rollouts.experiment_metadata(0.0, **config)
                metadata['git_revision'] = revision
                rollouts.save_cache(source, metadata, sig_bits=np.array([3, 52]),
                                    n_values=np.array([6]), sig_max=np.full((2, 1), value))

            argv = ['noise_vs_chop.py', '--rounding-cache', str(source), '--save', str(output),
                    '--num-samples', '10', '--dirs', '4', '--noise-scale', '0', '--no-noise']
            snapshots = []

            def capture(fig, path, metadata):
                snapshots.append((float(fig.axes[0].lines[0].get_ydata()[0]),
                                  metadata['rounding_source']['git_revision']))

            def run(*extra):
                with patch.object(sys, 'argv', argv + list(extra)), \
                        patch('core.matrices.matrix_family', return_value=lambda n: np.ones((2, 6))), \
                        patch.object(rollouts, 'run_samples', side_effect=self.fake_samples), \
                        patch.object(rollouts, 'save_figure', side_effect=capture):
                    noise_vs_chop.main()

            write_source('revision-A', 1.0)
            run()
            write_source('revision-B', 9.0)
            run('--plot-only')
            self.assertEqual(snapshots[1], (9.0, 'revision-B'))


if __name__ == '__main__':
    unittest.main()
