// Device-integration example: wire the detector into a streaming audio
// front end. The detector works on hop-sized advances over an FFT-sized
// sliding window, so the integration keeps a ring buffer and hands the
// detector a pointer to one full FFT window each hop. When a block is
// active, that window's samples are zeroed before reaching the ASR stage.
// Calibration runs once on a known-clean ambient segment at boot.
#include "detector.hpp"
#include "synth.hpp"
#include <cstdio>
#include <memory>
#include <vector>

using namespace hvc;

class FrontEndGuard {
public:
    FrontEndGuard(int sampleRate)
        : det_(std::make_unique<HiddenCommandDetector>()),
          fft_(det_->config().fftSize), hop_(det_->config().hop),
          ring_(fft_, 0.0f) {
        (void)sampleRate;  // detector config already fixed at 48 kHz
    }

    // Feed a clean ambient buffer once at startup, through the same mic
    // profile the live path uses.
    void calibrate(const std::vector<float>& ambient) {
        std::vector<float> padded(fft_, 0.0f);
        padded.insert(padded.end(), ambient.begin(), ambient.end());
        for (size_t p = 0; p + hop_ <= padded.size(); p += hop_)
            det_->calibrateFrame(&padded[p]);
        det_->finishCalibration();
    }

    // Append hop samples of incoming PCM, run one detector window, and mute
    // the newest hop in place when a block is active.
    bool pushHop(const float* hopSamples) {
        // slide the ring left by one hop, append the new samples
        for (int i = 0; i < fft_ - hop_; ++i) ring_[i] = ring_[i + hop_];
        for (int i = 0; i < hop_; ++i) ring_[fft_ - hop_ + i] = hopSamples[i];
        bool block = det_->processFrame(ring_.data());
        if (block)
            for (int i = fft_ - hop_; i < fft_; ++i) ring_[i] = 0.0f;
        return block;
    }

    void newUtterance() { det_->resetRuntime(); }

private:
    std::unique_ptr<HiddenCommandDetector> det_;
    int fft_, hop_;
    std::vector<float> ring_;
};

int main() {
    constexpr double fs = 48000.0;
    FrontEndGuard guard((int)fs);

    Rng r(0x1234);
    // 1 s ambient calibration through the same mic curve used for live audio.
    auto ambient = applyMic(whiteNoise(r, (size_t)fs, 0.004f), fs, 1, 0x11);
    guard.calibrate(ambient);

    auto attack = applyMic(ultrasonicAttack(r, fs, 4.0, 0.15f, 20000),
                           fs, 1, 0x22);
    auto benign = applyMic(farFieldSpeech(r, fs, 4.0), fs, 1, 0x33);
    const int hop = 512;

    for (const auto* tag : {"attack", "benign"}) {
        guard.newUtterance();
        const auto& sig = tag[0] == 'a' ? attack : benign;
        int blockedFrames = 0, total = 0;
        for (size_t p = 0; p + hop <= sig.size(); p += hop) {
            if (guard.pushHop(&sig[p])) ++blockedFrames;
            ++total;
        }
        std::printf("%-8s: %d/%d frames blocked (%.0f%%)\n", tag,
                    blockedFrames, total, 100.0 * blockedFrames / total);
    }
    return 0;
}
