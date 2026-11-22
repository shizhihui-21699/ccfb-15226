// Audio synthesis for detector evaluation. All signals are generated, no
// external data is needed. Each generator models the *physical* hallmarks of
// the attack or benign class rather than just playing a tone.
#pragma once
#include "dsp.hpp"
#include <string>

namespace hvc {

struct AudioClip {
    std::string label;     // class label
    int micId;             // which mic curve was applied
    std::vector<float> pcm;
};

// ---- building blocks -------------------------------------------------------

std::vector<float> whiteNoise(Rng& rng, size_t n, float rms);
// 1/f-ish pink noise via one-pole filtered white noise
std::vector<float> pinkNoise(Rng& rng, size_t n, float rms);

// Pseudo speech: voiced glottal pulses through 3 moving formants, plus
// unvoiced fricatives (band-limited noise bursts). Carries real syllable-rate
// amplitude modulation (4-10 Hz) and formant motion.
std::vector<float> pseudoSpeech(Rng& rng, double fs, float rms,
                                int syllables, bool telephoneBand = false);

// Pseudo music: chord pad (multiple detuned oscillators with vibrato),
// melodic note sequence, plus a soft percussive layer. Spectrally much flatter
// over time than speech (steady chord energy), with ~1-4 Hz beat.
std::vector<float> pseudoMusic(Rng& rng, double fs, float rms, double seconds);

void scaleRms(std::vector<float>& x, float targetRms);
float rmsOf(const std::vector<float>& x);
std::vector<float> mixAt(const std::vector<float>& a,
                         const std::vector<float>& b, double gainB);

// ---- attacks ---------------------------------------------------------------

// In-band hidden command: music bed + command at commandBelowMusicDb
// (command RMS n dB below music RMS). replayBandHz models the loudspeaker /
// injection path bandwidth (telephone-grade 3.4 kHz or small speaker ~5.5
// kHz); soft clipping models its nonlinearity. A real human voice in the
// room keeps its natural 5-7.5 kHz frication content.
std::vector<float> commanderSongClip(Rng& rng, double fs, double seconds,
                                     float musicRms, double commandBelowMusicDb,
                                     double replayLowHz, double replayBandHz);

// Ultrasonic carrier modulated by a speech envelope. The envelope is derived
// from an actual pseudo-speech waveform (full-wave rectified + lowpass), so
// the 1-12 Hz syllable modulation of real speech rides on an inaudible
// 17-23 kHz carrier. Multiple sub-carriers are used to widen the "formants"
// in the ultrasonic band, like real modulation-based attacks.
std::vector<float> ultrasonicAttack(Rng& rng, double fs, double seconds,
                                    float rms, double carrierHz);

// Steady ultrasonic tone/whine: same band, NO speech modulation. This is the
// adversarial benign case (electronic ballast, some switching supplies) that
// an energy-only detector would block.
std::vector<float> ultrasonicWhine(Rng& rng, double fs, double seconds,
                                   float rms);

// ---- benign scenes ---------------------------------------------------------

std::vector<float> farFieldSpeech(Rng& rng, double fs, double seconds);
std::vector<float> speechWithMusic(Rng& rng, double fs, double seconds,
                                   double speechAboveMusicDb);
std::vector<float> musicOnly(Rng& rng, double fs, double seconds);
std::vector<float> noiseOnly(Rng& rng, double fs, double seconds);
// Speech + mild reverberation-like echo path, low SNR far field.
std::vector<float> farFieldNoisy(Rng& rng, double fs, double seconds);

// ---- microphone emulation --------------------------------------------------

// Applies a mic-specific frequency response curve + broadband noise floor.
// Curve shapes: 0 = flat reference, 1 = rolled-off highs (MEMS cheap),
// 2 = peaked around 12k then steep rolloff (typical electret),
// 3 = boosted low + ripple. All curves differ above 15 kHz so the detector's
// response normalization matters.
std::vector<float> applyMic(std::vector<float> x, double fs, int micId,
                            uint64_t noiseSeed);

// ---- IO --------------------------------------------------------------------
bool writeWav(const std::string& path, const std::vector<float>& pcm,
              int sampleRate);

} // namespace hvc
