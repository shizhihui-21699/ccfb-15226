// Minimal real-time DSP primitives: radix-2 FFT, windows, biquad filters,
// mel filterbank, small math helpers. Pure C++17, no dependencies.
#pragma once
#include <cmath>
#include <complex>
#include <vector>
#include <cstddef>
#include <algorithm>

namespace hvc {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t nextU64() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
    }
    double uniform() { return (nextU64() >> 11) * (1.0 / 9007199254740992.0); }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }
    // Box-Muller
    double gaussian() {
        double u1 = std::max(uniform(), 1e-12), u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2);
    }
};

// Iterative in-place radix-2 Cooley-Tukey FFT. Size must be a power of two.
inline void fft(std::vector<std::complex<float>>& a) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = kTwoPi / (double)len;
        std::complex<float> wlen((float)std::cos(ang), (float)std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                auto u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

inline void hannWindow(std::vector<float>& w) {
    const size_t n = w.size();
    for (size_t i = 0; i < n; ++i)
        w[i] = (float)(0.5 - 0.5 * std::cos(kTwoPi * i / (n - 1)));
}

// RBJ biquad (transposed direct form II). Coefficients passed in B0..B2,A1,A2.
struct Biquad {
    float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    float z1 = 0, z2 = 0;
    void reset() { z1 = z2 = 0; }
    float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
    // Resonator (formant) with center f and bandwidth bw (Hz).
    static Biquad resonator(double fs, double f, double bw) {
        double r = std::exp(-kPi * bw / fs);
        double theta = kTwoPi * f / fs;
        Biquad q;
        q.b0 = (float)(1.0 - r); q.b1 = 0; q.b2 = 0;
        q.a1 = (float)(-2.0 * r * std::cos(theta));
        q.a2 = (float)(r * r);
        return q;
    }
    static Biquad bandpass(double fs, double f, double q) {
        double w0 = kTwoPi * f / fs, alpha = std::sin(w0) / (2.0 * q);
        double cosw = std::cos(w0);
        Biquad bq;
        double norm = 1.0 / (1.0 + alpha);
        bq.b0 = (float)(alpha * norm); bq.b1 = 0; bq.b2 = (float)(-alpha * norm);
        bq.a1 = (float)(-2.0 * cosw * norm); bq.a2 = (float)((1.0 - alpha) * norm);
        return bq;
    }
    static Biquad lowpass(double fs, double f) {
        double w0 = kTwoPi * f / fs, alpha = std::sin(w0) / 2.0;
        double cosw = std::cos(w0);
        Biquad bq;
        double norm = 1.0 / (1.0 + alpha);
        bq.b0 = (float)((1.0 - cosw) / 2.0 * norm);
        bq.b1 = (float)((1.0 - cosw) * norm);
        bq.b2 = (float)((1.0 - cosw) / 2.0 * norm);
        bq.a1 = (float)(-2.0 * cosw * norm);
        bq.a2 = (float)((1.0 - alpha) * norm);
        return bq;
    }
    static Biquad highpass(double fs, double f) {
        double w0 = kTwoPi * f / fs, alpha = std::sin(w0) / 2.0;
        double cosw = std::cos(w0);
        Biquad bq;
        double norm = 1.0 / (1.0 + alpha);
        bq.b0 = (float)((1.0 + cosw) / 2.0 * norm);
        bq.b1 = (float)(-(1.0 + cosw) * norm);
        bq.b2 = (float)((1.0 + cosw) / 2.0 * norm);
        bq.a1 = (float)(-2.0 * cosw * norm);
        bq.a2 = (float)((1.0 - alpha) * norm);
        return bq;
    }
};

inline double hzToMel(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
inline double melToHz(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// Triangular mel filterbank: list of (startBin, peakBin, endBin) per band.
struct MelBank {
    int nBands;
    std::vector<int> start, peak, end;
    MelBank() : nBands(0) {}
    MelBank(int bands, double fs, int fftSize, double loHz, double hiHz)
        : nBands(bands) {
        double melLo = hzToMel(loHz), melHi = hzToMel(hiHz);
        std::vector<double> pts(nBands + 2);
        for (int i = 0; i < nBands + 2; ++i)
            pts[i] = melToHz(melLo + (melHi - melLo) * i / (nBands + 1));
        start.resize(nBands); peak.resize(nBands); end.resize(nBands);
        for (int m = 0; m < nBands; ++m) {
            start[m] = std::max(0, (int)std::floor(pts[m] * fftSize / fs));
            peak[m] = (int)std::round(pts[m + 1] * fftSize / fs);
            end[m] = std::min(fftSize / 2, (int)std::ceil(pts[m + 2] * fftSize / fs));
        }
    }
    // energies from one-sided PSD (length fftSize/2+1)
    std::vector<float> apply(const std::vector<float>& psd) const {
        std::vector<float> e(nBands, 0.0f);
        for (int m = 0; m < nBands; ++m) {
            float acc = 0;
            for (int k = start[m]; k < end[m]; ++k) {
                float w;
                if (k <= peak[m])
                    w = (float)(k - start[m]) / std::max(1, peak[m] - start[m]);
                else
                    w = (float)(end[m] - k) / std::max(1, end[m] - peak[m]);
                acc += w * psd[k];
            }
            e[m] = acc;
        }
        return e;
    }
};

inline double dbSafe(double x) { return 10.0 * std::log10(std::max(x, 1e-12)); }
inline float dbSafeF(float x) { return (float)dbSafe(x); }

// Percentile on a copied vector (small arrays only).
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (v.size() - 1);
    size_t lo = (size_t)idx;
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

inline double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

// Zero-phase FFT filter of a whole signal with a given magnitude gain curve.
// gainAt(freqHz) is sampled on the FFT grid; signal is zero-padded to pow2.
inline std::vector<float> fftFilter(const std::vector<float>& x, double fs,
                                    const std::function<float(double)>& gainAt) {
    size_t n = 1;
    while (n < x.size()) n <<= 1;
    std::vector<std::complex<float>> spec(n);
    for (size_t i = 0; i < x.size(); ++i) spec[i] = x[i];
    fft(spec);
    for (size_t k = 0; k < n; ++k) {
        double f = k * fs / n;
        if (k > n / 2) f = -(double)(n - k) * fs / n;
        spec[k] *= gainAt(std::fabs(f));
    }
    // inverse via conjugate trick
    for (auto& c : spec) c = std::conj(c);
    fft(spec);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = std::conj(spec[i]).real() / (float)n;
    return y;
}

} // namespace hvc
