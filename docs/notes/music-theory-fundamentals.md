# Music Theory Fundamentals for AI Music Research

[中文版](music-theory-fundamentals-zh.md)

This introduction connects pitch, harmony, rhythm, texture, and form to music information retrieval (MIR) and generation. Its harmony examples describe Western tonal practice; they are not universal rules for all musical traditions. Definitions and illustrative calculations were reviewed on 2026-09-19.

## 1. Pitch

### 1.1 Frequency and tuning

Pitch is a perceptual attribute. For a periodic musical tone it usually corresponds to the fundamental frequency, even when the fundamental component is absent from the spectrum. Noise and inharmonic sounds may have ambiguous pitch. See the [UNSW acoustics explanation](https://phys.unsw.edu.au/jw/musFAQ.html).

In twelve-tone equal temperament (12-TET), an octave has frequency ratio 2:1 and contains 12 equal semitone steps. With A4 = 440 Hz and MIDI note number $m$:

$$
f(m)=440\,2^{(m-69)/12},\qquad m=69+12\log_2(f/440).
$$

Thus MIDI 69 is 440 Hz and MIDI 60 is approximately 261.63 Hz. A transposition by $n$ semitones multiplies frequency by $2^{n/12}$. This assumes the stated tuning: instruments can use other reference pitches, pitch bends, or alternative temperaments. See [UNSW's frequency/MIDI conversion](https://phys.unsw.edu.au/jw/notes.html).

### 1.2 Pitch class, octave, and spelling

Under the convention C4 = middle C, MIDI 0–127 spans C−1–G9. Software sometimes displays different octave numbers, so exchange MIDI numbers or frequencies when ambiguity matters. A pitch class groups pitches an octave apart; in a 12-class representation it is $m\bmod12$, with C = 0.

C♯ and D♭ are enharmonically equivalent in 12-TET. Their spelling can encode different harmonic functions. A chroma vector discards octave and spelling; its 12 bins typically aggregate spectral energy rather than necessarily counting notes. These are useful invariances for retrieval and real limitations for notation or voice-leading analysis.

## 2. Intervals and scales

### 2.1 Intervals

An interval has both a notated degree and a quality. Semitone count alone cannot determine the spelling: C–F♯ is an augmented fourth, while C–G♭ is a diminished fifth.

| Semitones in 12-TET | Common interval spelling |
|---|---|
| 0 | Perfect unison (P1) |
| 1 / 2 | Minor / major second (m2 / M2) |
| 3 / 4 | Minor / major third (m3 / M3) |
| 5 | Perfect fourth (P4) |
| 6 | Augmented fourth / diminished fifth (A4 / d5), tritone |
| 7 | Perfect fifth (P5) |
| 8 / 9 | Minor / major sixth (m6 / M6) |
| 10 / 11 | Minor / major seventh (m7 / M7) |
| 12 | Perfect octave (P8) |

The third and seventh of a **dominant seventh** chord form a tritone. This does not hold for every seventh-chord quality. Consonance and tension also depend on voicing, timbre, musical context, and listening experience.

### 2.2 Major, minor, and modes

The step patterns below run from tonic to the next octave; 1 means a semitone and 2 a whole tone.

| Scale | Successive steps | Example on A |
|---|---|---|
| Major | 2, 2, 1, 2, 2, 2, 1 | A B C♯ D E F♯ G♯ A |
| Natural minor | 2, 1, 2, 2, 1, 2, 2 | A B C D E F G A |
| Harmonic minor | 2, 1, 2, 2, 1, 3, 1 | A B C D E F G♯ A |
| Ascending melodic minor | 2, 1, 2, 2, 2, 2, 1 | A B C D E F♯ G♯ A |

Classical melodic-minor exercises normally descend using natural minor; jazz melodic minor commonly keeps the raised sixth and seventh in both directions. Raising the seventh in harmonic minor supports V and vii°7: in A minor, E–G♯–B and G♯–B–D–F. The seventh chord on ii is **B–D–F–A, half-diminished**, not fully diminished.

The diatonic modes are Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, and Locrian. Compare them using the **same tonic**: Dorian differs from natural minor by its raised sixth; Phrygian has a lowered second; Lydian differs from major by its raised fourth. Mood descriptions are associations rather than definitions. See [Open Music Theory's scale material](https://openmusictheory.github.io/scales.html).

### 2.3 Other pitch collections

- **Pentatonic** means five pitches per octave, not necessarily a scale without semitones. Major pentatonic uses degrees 1, 2, 3, 5, 6; minor pentatonic uses 1, ♭3, 4, 5, ♭7.
- **Whole-tone** uses six pitches separated by whole tones.
- **Octatonic/diminished** alternates semitones and whole tones, giving eight pitch classes.
- A common **minor blues scale** is 1, ♭3, 4, ♭5, 5, ♭7. Performed blue notes and bends need not sit exactly on this notated grid.

Scale masks can constrain a generator, but cannot by themselves represent a tradition's melodic grammar, intonation, or ornamentation.

## 3. Harmony

### 3.1 Triads and seventh chords

Root-position tertian triads contain a root, third, and fifth. The following offsets are measured in semitones above the root in 12-TET.

| Quality | Offsets | Example |
|---|---|---|
| Major | 0, 4, 7 | C E G |
| Minor | 0, 3, 7 | C E♭ G |
| Diminished | 0, 3, 6 | C E♭ G♭ |
| Augmented | 0, 4, 8 | C E G♯ |
| Major seventh | 0, 4, 7, 11 | C E G B |
| Dominant seventh | 0, 4, 7, 10 | C E G B♭ |
| Minor seventh | 0, 3, 7, 10 | C E♭ G B♭ |
| Half-diminished seventh | 0, 3, 6, 10 | C E♭ G♭ B♭ |
| Fully diminished seventh | 0, 3, 6, 9 | C E♭ G♭ B𝄫 |

The enharmonic spelling B𝄫 matters: it is the diminished seventh above C. Chord quality is distinct from function; a dominant-seventh-quality chord need not act as V in every style. See [Open Music Theory: triads and seventh chords](https://openmusictheory.github.io/triads.html).

Ninths, elevenths, and thirteenths are compound intervals whose pitch classes correspond to seconds, fourths, and sixths. Extensions also occur on minor chords: Dm9 is D–F–A–C–E. Their selection and voicing depend on context; there is no rule banning ninths on minor-seventh chords.

### 3.2 Functional harmony and cadences

In common-practice tonal music, I/i provides tonic function, ii/IV often provide predominant function, and V/vii° often provide dominant function. A common trajectory is I–IV–V–I. The roles of iii and vi depend on context. V7 often resolves to I/i, but deceptive resolutions, prolongation, and other continuations also occur. A V–I count is therefore not a universal quality metric.

| Cadence | Typical tonal realization |
|---|---|
| Perfect authentic (PAC) | V–I (or V–i), both in root position, tonic in the final soprano |
| Imperfect authentic (IAC) | An authentic cadence that does not meet the PAC conditions; terminology varies by textbook |
| Half (HC) | Phrase ends on V |
| Deceptive | V leads to an expected-tonic substitute, commonly vi in major or VI in minor |
| Plagal | IV–I, or iv–i in minor |

A progression becomes a cadence through phrase-ending context; every V–I is not automatically a cadence. See [Open Music Theory: cadence types](https://openmusictheory.github.io/cadenceTypes).

### 3.3 Voice leading

Voice leading tracks each part across chords. In a C-major G7–C resolution, B typically rises to C and F falls to E, both by semitone. G can remain as a common tone; D may move to C or E. The tendency tones behave differently in minor: in G7–Cm, F–E♭ is a whole step. See [Puget Sound: voice leading seventh chords](https://musictheory.pugetsound.edu/mt21c/VoiceLeadingSeventhChords.html).

Avoiding parallel perfect fifths and octaves is a convention of particular contrapuntal styles. It is not a general ban applicable to rock power chords, orchestral doubling, or parallel-chord textures. Evaluate voice leading against the intended style and preserve voice identity in the representation.

## 4. Rhythm, meter, and tempo

Whole, half, quarter, eighth, and sixteenth notes have relative durations 1, 1/2, 1/4, 1/8, and 1/16. A dot adds half the undotted duration. Seconds depend on the tempo and its beat unit: at quarter note = 120 BPM, a quarter lasts 0.5 seconds.

**Simple meter** divides each beat into two; examples include 2/4, 3/4, 4/4, 2/2, and 3/8. **Compound meter** divides each beat into three: 6/8 normally has two dotted-quarter beats, 9/8 three, and 12/8 four. The numerator counts eighth-note subdivisions in these examples, not groups of three. Additive groupings such as 7/8 = 2+2+3 require their own grouping information. See [Open Music Theory: meter](https://openmusictheory.github.io/meter.html).

In a basic 4/4 metrical hierarchy, beat 1 is strongest, beat 3 secondary, and beats 2 and 4 weaker; a backbeat can emphasize 2 and 4 against that hierarchy. Notation and performed accents are separate signals.

A **tuplet** specifies a nonstandard subdivision; a triplet is one kind, commonly three notes in the duration of two of the same nominal value. In 6/8, ordinary three-eighth-note groups do not require triplet notation. **Syncopation** accents or sustains events against expected metric accents. **Swing** involves unequal subdivisions and timing relationships that vary with tempo, performer, and context; 2:1 is an illustration, not a fixed standard.

Tempo terms such as adagio, andante, allegro, and presto describe broad tempo/character categories. Their BPM boundaries are not universal. Store an explicit beat unit with BPM, retain expressive timing where needed, and report half-/double-tempo ambiguities in beat tracking.

## 5. Melody, expectation, and structure

A melodic contour abstracts upward, downward, and repeated pitches; useful shapes include ascending, descending, arching, and undulating. Interval statistics depend on corpus and representation. A preference for small intervals does not establish a universal Zipf law.

A **motif** is a recognizable musical idea, possibly rhythmic as well as pitched. It has no fixed 2–8-note length. It can develop through sequence, inversion, retrograde, augmentation, and diminution. A **phrase** is a perceived musical unit; four- and eight-bar patterns are common in some repertoires, not required lengths.

Expectation contributes to tension and release alongside rhythm, register, dynamics, and context. Pitch/interval entropy measures distributional uncertainty, not boredom, creativity, or musical quality directly. Random sequences can have high entropy; meaningful repetition can have low entropy.

## 6. Texture, instruments, and sound

| Texture | Definition |
|---|---|
| Monophonic | A single melodic line, possibly doubled in unison/octaves |
| Homophonic | A principal melody with accompaniment, or parts moving mainly together |
| Polyphonic | Multiple relatively independent melodic parts |
| Heterophonic | Simultaneous variants of a shared melody |

An ensemble or cultural tradition can use several textures. A string quartet is not automatically polyphonic, nor is all Chinese music heterophonic.

A standard 88-key piano spans A0–C8 (MIDI 21–108). Distinguish **written** from **sounding** pitch: a B♭ clarinet/trumpet normally sounds a major second below its written pitch; a horn in F normally sounds a perfect fifth below. Instrument ranges depend on model, technique, and player. MIDI playback generally already represents sounding pitch; do not apply an additional transposition solely because a program is labeled “clarinet.” Score-to-MIDI conversion must account for the score's convention.

Timbre involves spectral shape, transients, noise, and evolution over time. An ADSR envelope is a synthesis model: attack/decay/release are times, while sustain is a level maintained during a held note. Strings and winds can have either sharp or gradual attacks depending on articulation.

Mixing labels such as bass, midrange, and presence have approximate, overlapping boundaries. Instruments and voices occupy overlapping frequency bands; source separation is not equivalent to frequency-band filtering. Removing frequencies above 20 kHz is filtering, not mathematically lossless compression, and computational savings depend on the actual resampling/model design.

## 7. Psychoacoustics

Auditory frequency selectivity and masking help explain perception and motivate perceptual audio models. Bark critical-band and ERB auditory-filter scales are related but distinct; fixed numbers of filters in an implementation are modeling choices. Masking thresholds and time spans depend on signal level, spectrum, duration, and the listener. Mel filterbanks approximate perceptual frequency spacing; they do not directly implement a full masking model.

For music generation, use perceptual metrics within their validated domain and supplement them with listening tests. A codec quality score does not establish harmonic or structural quality. See the [evaluation notes](music-evaluation.md).

## 8. Musical form

| Form | Schematic example |
|---|---|
| Binary | A–B |
| Ternary | A–B–A |
| Verse–chorus | Verse–chorus–verse–chorus–bridge–chorus |
| AABA | A–A–B–A; a 32-bar example has four eight-bar sections |
| Rondo | A–B–A–C–A |
| Sonata form | Exposition–development–recapitulation; introductions/codas are possible |

These are schemata with many variants. A chorus need not end with a PAC, a verse need not end with a half cadence, and a bridge need not end deceptively. Section labels should combine lyrics, melody, instrumentation, repetition, and harmonic context.

## 9. Notation and encoding in AI systems

| Representation | Useful properties | Main caveat |
|---|---|---|
| MIDI-like event sequence | Note events, timing, velocity, controllers | Meter/voice/spelling information depends on the encoding |
| REMI-style tokens | Explicit bar/position and optional tempo/chord information | Quantization and chord-estimator errors can propagate |
| Compound tokens | Group note/event attributes to shorten sequences | Attribute alignment and simultaneous-event ordering need specification |
| Piano roll | Pitch × time activation grid | Time quantization; articulation and repeated-note encoding need care |
| ABC | Readable textual notation with voices and other notation features | Supported features vary by parser; audio timbre is not represented |

Neural **audio** tokens do not inherently assume 12-TET: they encode sound and can retain continuous intonation. Note-only pitch-class encodings impose that restriction unless extended. MIDI also has pitch-bend and tuning mechanisms.

Key estimation often compares an observed pitch-class profile with rotated major/minor templates; learned systems can exploit temporal context. Krumhansl–Schmuckler-style methods use tonal profiles rather than a universal binary template or interval histogram. Binary templates are a separate baseline choice. Report dataset, tuning, key vocabulary, windowing, and scoring protocol; there is no general “90% for piano / 60–70% for orchestra” accuracy guarantee.

## 10. Further reading

- [Open Music Theory](https://viva.pressbooks.pub/openmusictheory/) — open textbook covering fundamentals, tonal analysis, rhythm, and other repertoires.
- [Music Theory for the 21st-Century Classroom](https://musictheory.pugetsound.edu/mt21c/) — theory explanations and worked examples by Robert Hutchinson.
- Meinard Müller, [*Fundamentals of Music Processing*, second edition](https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP) (2021) — author-maintained textbook site and computational examples.
- [Survey reading guide](../surveys/reading-guide.md) and [verified reference list](../../references/README.md).
