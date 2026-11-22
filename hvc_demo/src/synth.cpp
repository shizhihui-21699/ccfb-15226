#include "synth.hpp"
#include <fstream>
#include <cstdint>
#include <functional>

namespace hvc {

float rmsOf(const std::vector<float>& x) {
    double acc = 0;
    for (float v : x) acc += (double)v * v;
    return (float)std::sqrt(acc / std::max(size_t(1), x.size()));
}

void scaleRms(std::vector<float>& x, float targetRms) {
    float r = rmsOf(x);
    float g = targetRms / std::max(r, 1e-9f);
    for (auto& v : x) v *= g;
}

std::vector<float> whiteNoise(Rng& rng, size_t n, float rms) {
    std::vector<float> x(n);
    for (auto& v : x) v = (float)rng.gaussian();
    scaleRms(x, rms);
    return x;
}

std::vector<float> pinkNoise(Rng& rng, size_t n, float rms) {
    std::vector<float> x(n);
    float lp = 0;
    for (size_t i = 0; i < n; ++i) {
        float w = (float)rng.gaussian();
        lp = 0.96f * lp + 0.04f * w;          // one-pole -> ~1/f slope
        x[i] = 0.6f * w + lp;
    }
    scaleRms(x, rms);
    return x;
}

// Vowel-like formant targets (F1, F2, F3) in Hz.
static const double kFormants[][3] = {
    {270, 2290, 3010}, {530, 1840, 2480}, {730, 1090, 2440},
    {570,  840, 2410}, {300,  870, 2240}, {490, 1480, 2450},
};

std::vector<float> pseudoSpeech(Rng& rng, double fs, float rms,
                                int syllables, bool telephoneBand) {
    std::vector<float> sig;
    double f0 = rng.uniform(105, 175);
    int vi = (int)rng.uniform(0, 6);
    for (int s = 0; s < syllables; ++s) {
        double dur = rng.uniform(0.16, 0.34);
        size_t n = (size_t)(dur * fs);
        std::vector<float> syl(n, 0.0f);
        bool voiced = rng.uniform() < 0.72;
        int nvi = (vi + (int)rng.uniform(1, 4)) % 6;
        double f1 = kFormants[nvi][0] * rng.uniform(0.94, 1.06);
        double f2 = kFormants[nvi][1] * rng.uniform(0.95, 1.05);
        double f3 = kFormants[nvi][2] * rng.uniform(0.96, 1.04);
        Biquad r1 = Biquad::resonator(fs, f1, 90);
        Biquad r2 = Biquad::resonator(fs, f2, 110);
        Biquad r3 = Biquad::resonator(fs, f3, 160);
        Biquad bp = Biquad::bandpass(fs, rng.uniform(3500, 6500), 1.2);

        double period = fs / f0;
        long nextPulse = 0;
        for (size_t i = 0; i < n; ++i) {
            float src;
            if (voiced) {
                src = 0;
                if ((long)i >= nextPulse) {            // jittered glottal pulse
                    src = 1.0f;
                    double jitter = 1.0 + rng.uniform(-0.03, 0.03);
                    nextPulse = (long)std::lround((i + 1) + period * jitter);
                }
            } else {
                src = (float)rng.gaussian() * 0.5f;    // frication noise
            }
            float y = r3.process(r2.process(r1.process(src)));
            if (!voiced) y = 0.9f * bp.process(src) + 0.15f * y;
            // syllable amplitude envelope: fast attack, slower decay
            double t = (double)i / n;
            double env = std::min(1.0, t / 0.12) *
                         std::min(1.0, (1 - t) / 0.18);
            syl[i] = y * (float)(0.25 + 0.75 * env);
            // chest/glottal component: sub-150 Hz energy that rides the
            // syllable envelope. Present in a live voice, removed by a small
            // speaker's low-frequency cutoff.
            if (voiced) {
                double fChest = f0 * 0.6;
                double ph = 2 * kPi * fChest * (i / fs);
                syl[i] += (float)(0.5 * (0.25 + 0.75 * env) *
                                  (std::sin(ph) + 0.4 * std::sin(2 * ph)));
            }
        }
        sig.insert(sig.end(), syl.begin(), syl.end());
        size_t gap = (size_t)(rng.uniform(0.02, 0.09) * fs);
        sig.insert(sig.end(), gap, 0.0f);
        vi = nvi;
    }
    if (telephoneBand) {
        Biquad lp = Biquad::lowpass(fs, 3400);
        for (auto& v : sig) v = lp.process(v);
    }
    scaleRms(sig, rms);
    return sig;
}

std::vector<float> pseudoMusic(Rng& rng, double fs, float rms, double seconds) {
    size_t n = (size_t)(seconds * fs);
    std::vector<float> x(n, 0.0f);
    // chord pad: root + major/minor triad, detuned oscillators, vibrato
    double root = 110.0 * std::pow(2.0, (int)rng.uniform(0, 7) / 12.0);
    double ratios[3] = {1.0, rng.uniform() < 0.5 ? 1.2 : 1.19, 1.5};
    double vibRate = rng.uniform(4.0, 6.5), vibDepth = rng.uniform(2.0, 5.0);
    for (int c = 0; c < 3; ++c) {
        double f = root * ratios[c];
        for (int d = -1; d <= 1; ++d) {
            double ff = f * (1.0 + d * 0.004);
            double g = 0.10;
            for (size_t i = 0; i < n; ++i) {
                double t = i / fs;
                double inst = 2 * kPi * ff * t +
                              vibDepth / vibRate * std::sin(2 * kPi * vibRate * t);
                x[i] += (float)(g * std::sin(inst));
            }
        }
    }
    // pentatonic melody on top, note every beat (~2 Hz)
    double pent[5] = {1.0, 9.0 / 8, 5.0 / 4, 3.0 / 2, 5.0 / 3};
    double beat = 0.5;
    for (double start = 0; start < seconds; start += beat) {
        double f = root * 2 * pent[(int)rng.uniform(0, 5)] *
                   (rng.uniform() < 0.3 ? 2.0 : 1.0);
        size_t a = (size_t)(start * fs), b = std::min(n, a + (size_t)(beat * fs));
        for (size_t i = a; i < b; ++i) {
            double t = (i - a) / fs;
            double env = std::min(1.0, t / 0.02) * std::exp(-3.5 * t);
            double ph = 2 * kPi * f * (i / fs);
            x[i] += (float)(0.22 * env * (std::sin(ph) + 0.3 * std::sin(2 * ph)));
        }
        // soft percussive hit: short lowpassed noise
        size_t plen = (size_t)(0.05 * fs);
        for (size_t i = a; i < std::min(n, a + plen); ++i) {
            double t = (i - a) / fs;
            x[i] += (float)(0.12 * std::exp(-60 * t) * rng.gaussian());
        }
        // hi-hat: high-passed noise burst, rich 6-12 kHz content, on every
        // second beat. Real music's high end comes from cymbals like this;
        // a band-limited replay path cannot reproduce it.
        size_t hat = (size_t)(0.03 * fs);
        float hz = 0;
        for (size_t i = a; i < std::min(n, a + hat); ++i) {
            double t = (i - a) / fs;
            float w = (float)rng.gaussian();
            hz = 0.85f * hz + 0.15f * w;            // one-pole high-pass
            float hp = w - hz;
            x[i] += (float)(0.10 * std::exp(-90 * t) * hp);
        }
    }
    // bright melody harmonics give the music natural energy up to ~7 kHz
    for (double start = 0; start < seconds; start += 1.0) {
        double f = root * 4 * pent[(int)rng.uniform(0, 5)];
        size_t a = (size_t)(start * fs), b = std::min(n, a + (size_t)(0.4 * fs));
        for (size_t i = a; i < b; ++i) {
            double t = (i - a) / fs;
            double env = std::min(1.0, t / 0.01) * std::exp(-6 * t);
            x[i] += (float)(0.06 * env * std::sin(2 * kPi * f * (i / fs)));
        }
    }
    scaleRms(x, rms);
    return x;
}

std::vector<float> mixAt(const std::vector<float>& a,
                         const std::vector<float>& b, double gainB) {
    size_t n = std::max(a.size(), b.size());
    std::vector<float> y(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        y[i] = (i < a.size() ? a[i] : 0.0f) +
               (float)(gainB * (i < b.size() ? b[i] : 0.0f));
    }
    return y;
}

// ---------------- attacks ----------------

std::vector<float> commanderSongClip(Rng& rng, double fs, double seconds,
                                     float musicRms, double commandBelowMusicDb,
                                     double replayLowHz, double replayBandHz) {
    auto bed = pseudoMusic(rng, fs, musicRms, seconds);
    int syl = std::max(4, (int)(seconds / 0.28));
    auto cmd = pseudoSpeech(rng, fs, 1.0f, syl);
    // Replay-channel model: a small speaker is a band-pass device. The high
    // cutoff removes fricatives; the low cutoff removes glottal/chest energy.
    Biquad hp = Biquad::highpass(fs, replayLowHz);
    Biquad lp = Biquad::lowpass(fs, replayBandHz);
    for (auto& v : cmd) v = lp.process(hp.process(v));
    // Soft clipping (tanh-like) of the small amplifier: adds odd harmonics
    // below the band cap, no low or high extension is regenerated.
    for (auto& v : cmd) v = std::tanh(1.6f * v) / std::tanh(1.6f);
    float cmdRms = musicRms * (float)std::pow(10.0, -commandBelowMusicDb / 20.0);
    scaleRms(cmd, cmdRms);
    // place command somewhere inside the clip with a small lead-in
    size_t lead = (size_t)(rng.uniform(0.15, 0.4) * fs);
    std::vector<float> placed(lead, 0.0f);
    placed.insert(placed.end(), cmd.begin(), cmd.end());
    if (placed.size() > bed.size()) bed.resize(placed.size(), 0.0f);
    auto y = mixAt(bed, placed, 1.0);
    y.resize(bed.size());
    return y;
}

std::vector<float> ultrasonicAttack(Rng& rng, double fs, double seconds,
                                    float rms, double carrierHz) {
    size_t n = (size_t)(seconds * fs);
    int syl = std::max(4, (int)(seconds / 0.3));
    auto speech = pseudoSpeech(rng, fs, 1.0f, syl);
    // speech envelope: rectify + one-pole lowpass (~120 Hz)
    std::vector<float> env(std::min(n, speech.size()));
    float lp = 0;
    float emax = 1e-9f;
    for (size_t i = 0; i < env.size(); ++i) {
        lp = 0.995f * lp + 0.005f * std::fabs(speech[i]);
        env[i] = lp;
        emax = std::max(emax, lp);
    }
    std::vector<float> x(n, 0.0f);
    // sub-carriers spread like modulated formants
    double offs[5] = {0, 700, -700, 1400, -1400};
    double wts[5]  = {1.0, 0.55, 0.5, 0.28, 0.25};
    for (size_t i = 0; i < n; ++i) {
        float e = (i < env.size() ? env[i] / emax : 0.05f);
        e = 0.25f + 0.75f * e;                          // residual carrier + AM
        double t = i / fs;
        double v = 0;
        for (int k = 0; k < 5; ++k) {
            double fc = carrierHz + offs[k] +
                        30.0 * std::sin(2 * kPi * 1.3 * t + k);
            v += wts[k] * std::cos(2 * kPi * fc * t);
        }
        x[i] = (float)(e * v / 2.6);
    }
    scaleRms(x, rms);
    return x;
}

std::vector<float> ultrasonicWhine(Rng& rng, double fs, double seconds,
                                   float rms) {
    size_t n = (size_t)(seconds * fs);
    double f1 = rng.uniform(17500, 22500);
    double f2 = f1 + rng.uniform(900, 1600);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        double t = i / fs;
        double vib = 12 * std::sin(2 * kPi * 0.7 * t);   // tiny drift, no AM
        x[i] = (float)(0.8 * std::cos(2 * kPi * (f1 + vib) * t) +
                       0.3 * std::cos(2 * kPi * f2 * t));
    }
    scaleRms(x, rms);
    return x;
}

// ---------------- benign scenes ----------------

std::vector<float> farFieldSpeech(Rng& rng, double fs, double seconds) {
    int syl = std::max(5, (int)(seconds / 0.3));
    auto sp = pseudoSpeech(rng, fs, 0.12f, syl);
    auto bg = pinkNoise(rng, sp.size(), 0.02f);          // ~15 dB SNR
    std::vector<float> y = mixAt(sp, bg, 1.0);
    // simple echo: one delayed, attenuated copy
    size_t d = (size_t)(0.035 * fs);
    std::vector<float> echo(y.size(), 0.0f);
    for (size_t i = d; i < y.size(); ++i) echo[i] = 0.28f * y[i - d];
    for (size_t i = 0; i < y.size(); ++i) y[i] += echo[i];
    if (y.size() < (size_t)(seconds * fs))
        y.resize((size_t)(seconds * fs), 0.0f);
    return y;
}

std::vector<float> farFieldNoisy(Rng& rng, double fs, double seconds) {
    int syl = std::max(5, (int)(seconds / 0.3));
    auto sp = pseudoSpeech(rng, fs, 0.08f, syl);         // quiet, far talker
    auto bg = pinkNoise(rng, (size_t)(seconds * fs), 0.03f);
    size_t d = (size_t)(0.05 * fs);
    std::vector<float> echo(sp.size(), 0.0f);
    for (size_t i = d; i < sp.size(); ++i) echo[i] = 0.35f * sp[i - d];
    auto y = mixAt(bg, sp, 1.0);
    for (size_t i = 0; i < y.size() && i < echo.size(); ++i) y[i] += echo[i];
    y.resize((size_t)(seconds * fs), 0.0f);
    return y;
}

std::vector<float> speechWithMusic(Rng& rng, double fs, double seconds,
                                   double speechAboveMusicDb) {
    auto bed = pseudoMusic(rng, fs, 0.15f, seconds);
    int syl = std::max(4, (int)(seconds / 0.3));
    auto sp = pseudoSpeech(rng, fs, 1.0f, syl);
    float target = 0.15f * (float)std::pow(10.0, speechAboveMusicDb / 20.0);
    scaleRms(sp, target);
    size_t lead = (size_t)(rng.uniform(0.1, 0.3) * fs);
    std::vector<float> placed(lead, 0.0f);
    placed.insert(placed.end(), sp.begin(), sp.end());
    if (placed.size() > bed.size()) bed.resize(placed.size());
    auto y = mixAt(bed, placed, 1.0);
    y.resize((size_t)(seconds * fs), 0.0f);
    return y;
}

std::vector<float> musicOnly(Rng& rng, double fs, double seconds) {
    return pseudoMusic(rng, fs, 0.15f, seconds);
}

std::vector<float> noiseOnly(Rng& rng, double fs, double seconds) {
    return pinkNoise(rng, (size_t)(seconds * fs), 0.05f);
}

// ---------------- microphone emulation ----------------

std::vector<float> applyMic(std::vector<float> x, double fs, int micId,
                            uint64_t noiseSeed) {
    auto curve = [&](double f) -> float {
        double g = 1.0;
        switch (micId) {
        case 0:                                         // reference, near flat
            g = 1.0 + 0.05 * std::sin(f / 1800.0);
            break;
        case 1:                                         // cheap MEMS: high rolloff
            g = 1.0 / (1.0 + std::pow(f / 9000.0, 2.2));
            g *= 0.75 + 0.25 / (1.0 + std::pow(f / 120.0, 2));
            g = std::max(g, 0.10);
            break;
        case 2: {                                        // electret: 12k peak
            double peak = 1.0 + 1.0 * std::exp(-std::pow((f - 12000) / 2500.0, 2));
            double roll = 1.0 / (1.0 + std::pow(f / 15500.0, 4.0));
            g = 0.9 * peak * roll;
            g = std::max(g, 0.08);
            break;
        }
        case 3:                                          // boosted lows + ripple
            g = (1.0 + 1.8 / (1.0 + std::pow(f / 220.0, 2))) *
                (1.0 + 0.12 * std::sin(f / 1100.0)) *
                (1.0 / (1.0 + std::pow(f / 17000.0, 3.0)));
            g = std::max(g, 0.12);
            break;
        }
        return (float)g;
    };
    auto y = fftFilter(x, fs, curve);
    Rng nr(noiseSeed);
    for (auto& v : y) v += (float)(4e-4 * nr.gaussian());  // self-noise floor
    return y;
}

// ---------------- WAV IO ----------------

bool writeWav(const std::string& path, const std::vector<float>& pcm,
              int sampleRate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t dataBytes = (uint32_t)(pcm.size() * 2);
    uint32_t chunk = 36 + dataBytes, rate = (uint32_t)sampleRate;
    uint16_t ch = 1, bps = 16;
    uint32_t byteRate = rate * ch * bps / 8;
    uint16_t blockAlign = ch * bps / 8;
    f.write("RIFF", 4); f.write((char*)&chunk, 4); f.write("WAVE", 4);
    f.write("fmt ", 4); uint32_t fmtLen = 16;
    f.write((char*)&fmtLen, 4); uint16_t pcmFmt = 1;
    f.write((char*)&pcmFmt, 2); f.write((char*)&ch, 2);
    f.write((char*)&rate, 4); f.write((char*)&byteRate, 4);
    f.write((char*)&blockAlign, 2); f.write((char*)&bps, 2);
    f.write("data", 4); f.write((char*)&dataBytes, 4);
    for (float v : pcm) {
        int s = (int)std::lround(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        int16_t q = (int16_t)s;
        f.write((char*)&q, 2);
    }
    return (bool)f;
}

} // namespace hvc
