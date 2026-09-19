"""Independent numerical/round-trip checks for the CPU cookbook examples."""
import ast
import re
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.audio_baselines import (aligned_snr, chord_templates, estimate_key,
                                 pitch_class_entropy, preprocess_audio)

class NumericalExamples(unittest.TestCase):
    def test_entropy_probability_normalization(self):
        self.assertAlmostEqual(pitch_class_entropy(np.ones((12, 4))*7), np.log2(12))
        one = np.zeros((12, 3)); one[9] = 10
        self.assertAlmostEqual(pitch_class_entropy(one), 0)
        self.assertTrue(np.isnan(pitch_class_entropy(np.zeros((12, 4)))))

    def test_pitch_class_mapping(self):
        labels, matrix = chord_templates()
        for label, expected in [('C:maj', [0,4,7]), ('G:maj', [2,7,11]), ('A:min', [0,4,9])]:
            np.testing.assert_array_equal(np.flatnonzero(matrix[:,labels.index(label)]), expected)
        self.assertIsNone(estimate_key(np.ones((12, 8)))[0])

    def test_known_transposed_key_profile(self):
        # A published C-major probe-tone profile shifted to D; correlation must be 1.
        profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        label, score = estimate_key(np.roll(profile, 2)[:,None])
        self.assertEqual(label, 'D:major'); self.assertAlmostEqual(score, 1)

    def test_snr_known_error(self):
        x = np.array([1.,-1.,1.,-1.])
        self.assertAlmostEqual(aligned_snr(x, .9*x), 20)
        self.assertTrue(np.isinf(aligned_snr(x, x)))
        with self.assertRaises(ValueError): aligned_snr(x, x[:2])

    def test_silent_and_stereo_preprocessing(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, samples in [('silence',np.zeros((8000,2))),
                                  ('tone',np.sin(2*np.pi*440*np.arange(8000)/8000)[:,None]*np.ones((1,2)))]:
                src = root/(name+'.wav'); dst=root/'new'/name/'result.wav'
                sf.write(src,samples,8000,subtype='FLOAT')
                y=preprocess_audio(src,dst,target_sr=16000,target_duration=1.5)
                self.assertEqual(y.shape,(24000,));self.assertTrue(np.isfinite(y).all())
                np.testing.assert_array_equal(y[16000:],np.zeros(8000))
                if name=='silence': self.assertEqual(float(np.max(np.abs(y))),0)
                else: self.assertAlmostEqual(float(np.max(np.abs(y))),.95,places=5)
                self.assertEqual(sf.info(dst).samplerate,16000)

    def test_synthetic_audio_features_and_stft(self):
        import librosa
        sr=22050;t=np.arange(sr*3)/sr;y=.1*np.sin(2*np.pi*440*t)
        stft=librosa.stft(y,n_fft=2048,hop_length=512)
        reconstructed=librosa.istft(stft,hop_length=512,length=len(y))
        np.testing.assert_allclose(y,reconstructed,atol=1e-12)
        mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
        self.assertEqual(mel.shape[0],128);self.assertTrue(np.isfinite(mel).all())
        cq=librosa.cqt(y=y,sr=sr,n_bins=84,bins_per_octave=12)
        chroma=librosa.feature.chroma_cqt(C=np.abs(cq),sr=sr,bins_per_octave=12)
        self.assertEqual(int(chroma.mean(axis=1).argmax()),9) # 440 Hz is A
        tempo, beats=librosa.beat.beat_track(y=y,sr=sr)
        self.assertTrue(np.isfinite(float(np.asarray(tempo).reshape(-1)[0])))

    def test_midi_token_round_trip(self):
        import pretty_midi
        from miditok import REMI,TokenizerConfig
        from symusic import Score
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.mid';out=Path(tmp)/'output.mid'
            midi=pretty_midi.PrettyMIDI(initial_tempo=120)
            instrument=pretty_midi.Instrument(program=0)
            instrument.notes=[pretty_midi.Note(80,p,i*.5,i*.5+.5) for i,p in enumerate([60,64,67])]
            midi.instruments.append(instrument);midi.write(str(path))
            tokenizer=REMI(TokenizerConfig(use_programs=True,use_tempos=True))
            tokens=tokenizer(Score(str(path)));tokenizer.decode(tokens).dump_midi(str(out))
            restored=pretty_midi.PrettyMIDI(str(out))
            self.assertEqual([n.pitch for n in restored.instruments[0].notes],[60,64,67])
            bpm=float(restored.get_tempo_changes()[1][0])
            # Default tempo bins are quantized: compare musical beats, not exact seconds.
            self.assertLess(abs(bpm-120), 3.5)
            for i,note in enumerate(restored.instruments[0].notes):
                self.assertAlmostEqual(note.start*bpm/60,i,places=5)
                self.assertAlmostEqual(note.end*bpm/60,i+1,places=5)

    def test_extreme_finite_inputs(self):
        uniform=np.ones((12,4))*1e300
        self.assertAlmostEqual(pitch_class_entropy(uniform),np.log2(12))
        self.assertEqual(estimate_key(np.eye(12)[:,[0]]*1e-100)[0],estimate_key(np.eye(12)[:,[0]])[0])
        self.assertAlmostEqual(aligned_snr(np.array([1e300,-1e300]),np.array([.9e300,-.9e300])),20)
        for bad in [np.full((12,4),np.nan),-np.ones((12,4))]:
            with self.assertRaises(ValueError):pitch_class_entropy(bad)

    def test_cookbook_python_syntax(self):
        text=(ROOT/'docs/cookbook/recipes.md').read_text()
        blocks=re.findall(r'```python\n(.*?)\n```',text,re.S)
        self.assertGreater(len(blocks),8)
        for i,block in enumerate(blocks): ast.parse(block,filename=f'cookbook-block-{i}')

if __name__=='__main__': unittest.main()
