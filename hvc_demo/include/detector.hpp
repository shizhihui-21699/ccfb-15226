// On-device streaming detector for hidden voice commands.
//
// Channel A - ultrasonic modulation (DolphinAttack-class):
//   Energy that dominates the 17-23.5 kHz band, normalized against the mic's
//   own per-bin noise floor (calibration), AND showing speech-rate (3-12 Hz)
//   amplitude modulation spread over several carriers. A steady ultrasonic
//   whine is rejected: it has no syllable-rate modulation and collapses onto
//   one or two narrow spectral peaks.
//
// Channel B - in-band replay/liveness gate (CommanderSong-class):
//   A voice-band signal with syllable-rate modulation riding on a tonal bed,
//   but WITHOUT either signature of a live human voice - chest/glottal
//   modulation below 150 Hz, and high-band fricative modulation. Small
//   replay speakers are band-pass devices and can produce neither. Decisions
//   use ~0.7 s window medians and a hard warmup to suppress startup noise.
//
// Every level feature is a ratio or a calibrated z-score, so absolute mic
// gain and high-frequency roll-off differences do not move the decision.
#pragma once
#include "dsp.hpp"
#include <deque>

namespace hvc {

struct DetectorConfig {
    int sampleRate = 48000;
    int fftSize = 1024;
    int hop = 512;                       // 10.67 ms per frame at 48 kHz

    // Channel A (ultrasonic)
    double usLowHz = 17000, usHighHz = 23500;
    double usFloorMarginDb = 12.0;
    double usRatioMin = 0.30;           // band dominated by ultrasonic energy
    double usAmDepthMin = 0.18;         // syllable-rate (3-12 Hz) modulation
    double usNarrowMin = 0.30;
    double usPeakinessMax = 0.88;       // reject a steady 1-2 tone whine
    int    usVoteFrames = 5, usVoteWindow = 8;
    int    usHoldFrames = 24;           // ~256 ms hysteresis hold

    // Channel B (in-band replay gate)
    double musicFlatnessMax = 0.20;     // tonal/music-like bed
    double modSyllMin = 0.07;           // syllable-rate modulation present
    double speechRatioMinB = 0.08;      // voice-band content present
    // Replay/live boundary on the ~0.7 s window medians. Conservative by
    // design: the front end must never block a real user or ordinary music,
    // so deeply masked commands that fall inside the music distribution are
    // left to the ASR-backend defense.
    double livenessMax = 0.09;          // chest/glottal syllable modulation
    double hfSyllMax = 0.18;            // high-band fricative modulation
    int    warmupFrames = 64;           // ~0.68 s, no B verdict before this
    int    bPersistFrames = 12, bWindow = 16;
    int    bHoldFrames = 24;

    int    cooldownFrames = 48;         // ~512 ms between block re-triggers
    int    melBands = 16;
};

struct FrameScore {
    // Channel A
    double usRatio = 0, usAboveFloor = 0, usAmDepth = 0, usPeakiness = 0;
    // Channel B
    double flatness = 0, modSyll = 0, speechRatio = 0;
    double livenessMod = 0, hfConsonant = 0;
    double liveMedian = 0, hfMedian = 0;
    bool   aRaw = false, bRaw = false;
};

struct BlockEvent { int frameIndex; char channel; };

class HiddenCommandDetector {
public:
    explicit HiddenCommandDetector(const DetectorConfig& cfg = {})
        : cfg_(cfg), mel_(cfg.melBands, cfg.sampleRate, cfg.fftSize, 100, 7500) {
        win_.resize(cfg_.fftSize);
        hannWindow(win_);
        spec_.resize(cfg_.fftSize);
        psd_.resize(cfg_.fftSize / 2 + 1);
        usLo_ = hz2bin(cfg_.usLowHz); usHi_ = hz2bin(cfg_.usHighHz);
        spLo_ = hz2bin(300); spHi_ = hz2bin(7500);
        floorMedian_.assign(psd_.size(), 0.0);
        floorMad_.assign(psd_.size(), 1.0);
        bandHistory_.resize(cfg_.melBands);
    }

    // Calibrate on a known-clean ambient segment, then freeze per-bin floors.
    void calibrateFrame(const float* frame) {
        analyze(frame);
        calPsd_.push_back(psd_);
    }
    void finishCalibration() {
        if (calPsd_.empty()) return;
        std::vector<double> col(calPsd_.size());
        for (size_t k = 0; k < psd_.size(); ++k) {
            for (size_t i = 0; i < calPsd_.size(); ++i)
                col[i] = std::log(std::max(calPsd_[i][k], 1e-12f));
            double med = median(col);
            std::vector<double> dev(col.size());
            for (size_t i = 0; i < col.size(); ++i) dev[i] = std::fabs(col[i] - med);
            floorMedian_[k] = med;
            floorMad_[k] = std::max(median(dev), 0.05);
        }
        calPsd_.clear();
        calibrated_ = true;
    }
    bool isCalibrated() const { return calibrated_; }

    // Process one hop-sized frame; true while a block is active.
    bool processFrame(const float* frame) {
        analyze(frame);
        FrameScore s;
        computeScores(s);
        scores_.push_back(s);
        return updateState(s);
    }

    int processStream(const std::vector<float>& pcm) {
        int events = 0;
        for (size_t pos = 0; pos + cfg_.hop <= pcm.size(); pos += cfg_.hop)
            if (processFrame(&pcm[pos]) && !blockActive_) ++events;
        return events;
    }

    bool blockActive() const { return blockActive_; }
    const std::vector<BlockEvent>& events() const { return events_; }
    const std::vector<FrameScore>& scores() const { return scores_; }
    const DetectorConfig& config() const { return cfg_; }

    void resetRuntime() {
        blockActive_ = false; cooldown_ = 0;
        aHold_ = bHold_ = 0; aVotes_.clear(); bVotes_.clear();
        usEnergy_.clear(); livenessEnergy_.clear(); hfEnergy_.clear();
        liveWin_.clear(); hfWin_.clear();
        liveMedian_ = hfMedian_ = 0.0;
        bFrameCount_ = 0;
        for (auto& q : bandHistory_) q.clear();
        scores_.clear(); events_.clear();
    }

private:
    int hz2bin(double hz) const {
        return (int)std::round(hz * cfg_.fftSize / cfg_.sampleRate);
    }

    void analyze(const float* frame) {
        for (int i = 0; i < cfg_.fftSize; ++i)
            spec_[i] = std::complex<float>(frame[i] * win_[i], 0.0f);
        fft(spec_);
        for (size_t k = 0; k < psd_.size(); ++k) {
            float re = spec_[k].real(), im = spec_[k].imag();
            psd_[k] = (re * re + im * im) / (cfg_.fftSize * cfg_.fftSize);
        }
    }

    double floorZ(int k) const {
        if (!calibrated_) return 0.0;
        double v = std::log(std::max(psd_[k], 1e-12f));
        return (v - floorMedian_[k]) / floorMad_[k];
    }

    // Fraction of an energy trajectory's temporal power in 3-12 Hz syllable
    // bins (frame rate = fs/hop).
    double syllableFraction(const std::vector<float>& hist) const {
        int n = (int)hist.size();
        if (n < 24) return 0.0;
        double mean = 0;
        for (float v : hist) mean += v;
        mean /= n;
        double syll = 0, total = 0;
        double rate = (double)cfg_.sampleRate / cfg_.hop;
        for (int b = 1; b <= n / 2; ++b) {
            double re = 0, im = 0;
            for (int t = 0; t < n; ++t) {
                double ang = -kTwoPi * b * t / n;
                double d = hist[t] - mean;
                re += d * std::cos(ang); im += d * std::sin(ang);
            }
            double pw = (re * re + im * im) / (double)(n * n);
            total += pw;
            if (b * rate / n >= 3.0 && b * rate / n <= 12.0) syll += pw;
        }
        return total > 1e-18 ? syll / total : 0.0;
    }

    void computeScores(FrameScore& s) {
        double total = 0, usE = 0, spE = 0;
        for (size_t k = 1; k < psd_.size(); ++k) total += psd_[k];
        for (int k = usLo_; k <= usHi_; ++k) usE += psd_[k];
        for (int k = spLo_; k <= spHi_; ++k) spE += psd_[k];
        total = std::max(total, 1e-12);
        s.usRatio = usE / total;
        s.speechRatio = spE / total;

        // ---- Channel A: floor-normalized level, peakiness, AM depth ----
        double above = 0, peakE = 0;
        int narrowBins = 0;
        double zThresh = cfg_.usFloorMarginDb / (10.0 * 0.4343);
        for (int k = usLo_; k <= usHi_; ++k) {
            if (floorZ(k) > zThresh) { above += psd_[k]; ++narrowBins; }
            peakE = std::max(peakE, (double)psd_[k]);
        }
        s.usAboveFloor = above / total;
        std::vector<double> band;
        for (int k = usLo_; k <= usHi_; ++k) band.push_back(psd_[k]);
        std::sort(band.begin(), band.end(), std::greater<double>());
        double top4 = 0;
        for (int i = 0; i < std::min(4, (int)band.size()); ++i) top4 += band[i];
        s.usPeakiness = usE > 1e-18 ? std::min(1.0, top4 / usE) : 1.0;

        usEnergy_.push_back((float)usE);
        if (usEnergy_.size() > 48) usEnergy_.erase(usEnergy_.begin());
        s.usAmDepth = syllableFraction(usEnergy_);

        s.aRaw = s.usRatio > cfg_.usRatioMin &&
                 s.usAboveFloor > cfg_.usRatioMin &&
                 s.usAmDepth > cfg_.usAmDepthMin &&
                 s.usPeakiness < cfg_.usPeakinessMax &&
                 narrowBins >= 3;

        // ---- Channel B: tonal bed, voice modulation, liveness gate ----
        auto bands = mel_.apply(psd_);
        double geo = 0, arith = 0;
        int M = cfg_.melBands;
        for (float e : bands) { geo += std::log(std::max((double)e, 1e-12)); arith += e; }
        geo = std::exp(geo / M); arith /= M;
        s.flatness = arith > 1e-12 ? std::min(1.0, geo / arith) : 0;

        for (int m = 0; m < M; ++m) {
            bandHistory_[m].push_back(bands[m]);
            if (bandHistory_[m].size() > 48) bandHistory_[m].pop_front();
        }
        double frameRate = (double)cfg_.sampleRate / cfg_.hop;
        double syllPow = 0, totalPow = 0;
        for (int m = 0; m < M; ++m) {
            const auto& q = bandHistory_[m];
            int n = (int)q.size();
            if (n < 16) continue;
            double mean = 0;
            for (float v : q) mean += v;
            mean /= n;
            if (mean < 1e-10) continue;
            for (int b = 1; b <= n / 2; ++b) {
                double re = 0, im = 0;
                for (int t = 0; t < n; ++t) {
                    double ang = -kTwoPi * b * t / n;
                    double d = q[t] - mean;
                    re += d * std::cos(ang); im += d * std::sin(ang);
                }
                double pw = (re * re + im * im) / (double)(n * n);
                totalPow += pw;
                if (b * frameRate / n >= 3.0 && b * frameRate / n <= 12.0)
                    syllPow += pw;
            }
        }
        s.modSyll = totalPow > 1e-12 ? std::min(1.0, syllPow / totalPow) : 0;

        // Liveness: chest/glottal energy 40-150 Hz syllable modulation.
        int lfLo = std::max(1, hz2bin(40)), lfHi = hz2bin(150);
        double lfE = 0;
        for (int k = lfLo; k <= lfHi; ++k) lfE += psd_[k];
        livenessEnergy_.push_back((float)lfE);
        if (livenessEnergy_.size() > 48) livenessEnergy_.erase(livenessEnergy_.begin());
        s.livenessMod = syllableFraction(livenessEnergy_);

        // Fricative: 5-7.5 kHz syllable modulation.
        int hfLo = hz2bin(5000);
        double hfE = 0;
        for (int k = hfLo; k <= spHi_; ++k) hfE += psd_[k];
        hfEnergy_.push_back((float)hfE);
        if (hfEnergy_.size() > 48) hfEnergy_.erase(hfEnergy_.begin());
        s.hfConsonant = syllableFraction(hfEnergy_);

        // Time-accumulated window medians.
        liveWin_.push_back((float)s.livenessMod);
        if (liveWin_.size() > 64) liveWin_.erase(liveWin_.begin());
        if (liveWin_.size() >= 32)
            liveMedian_ = median(std::vector<double>(liveWin_.begin(), liveWin_.end()));
        hfWin_.push_back((float)s.hfConsonant);
        if (hfWin_.size() > 64) hfWin_.erase(hfWin_.begin());
        if (hfWin_.size() >= 32)
            hfMedian_ = median(std::vector<double>(hfWin_.begin(), hfWin_.end()));
        s.liveMedian = liveMedian_;
        s.hfMedian = hfMedian_;
        ++bFrameCount_;

        bool warmed = bFrameCount_ >= cfg_.warmupFrames &&
                      liveMedian_ > 0.0 && hfMedian_ > 0.0;
        bool tonalBed = s.flatness < cfg_.musicFlatnessMax;
        bool speechMod = s.modSyll > cfg_.modSyllMin &&
                         s.speechRatio > cfg_.speechRatioMinB;
        // Replay gate on the time-accumulated window MEDIANS (not peaks):
        // a band-limited replay path sustains neither chest/glottal
        // modulation below 150 Hz nor high-band fricative modulation. Pure
        // music is rejected by the fricative gate (its cymbals keep the
        // high-band modulation median above the cap); a live user clears
        // both gates. The 0.68 s warmup + 12-of-16 frame vote make this
        // robust to per-frame noise.
        bool noLiveness = liveMedian_ < cfg_.livenessMax &&
                          hfMedian_ < cfg_.hfSyllMax;
        s.bRaw = warmed && tonalBed && speechMod && noLiveness;
    }

    bool updateState(const FrameScore& s) {
        aVotes_.push_back(s.aRaw);
        if ((int)aVotes_.size() > cfg_.usVoteWindow) aVotes_.pop_front();
        int ayes = 0;
        for (bool v : aVotes_) ayes += v;
        if (ayes >= cfg_.usVoteFrames) aHold_ = cfg_.usHoldFrames;
        else aHold_ = std::max(0, aHold_ - 1);

        bVotes_.push_back(s.bRaw);
        if ((int)bVotes_.size() > cfg_.bWindow) bVotes_.pop_front();
        int byes = 0;
        for (bool v : bVotes_) byes += v;
        if (byes >= cfg_.bPersistFrames) bHold_ = cfg_.bHoldFrames;
        else bHold_ = std::max(0, bHold_ - 1);

        bool fire = aHold_ > 0 || bHold_ > 0;
        if (cooldown_ > 0) --cooldown_;
        if (fire && (!blockActive_ || cooldown_ == 0)) {
            blockActive_ = true;
            cooldown_ = cfg_.cooldownFrames;
            events_.push_back({(int)scores_.size(), aHold_ > 0 ? 'A' : 'B'});
        } else if (!fire) {
            blockActive_ = false;
        }
        return blockActive_;
    }

    DetectorConfig cfg_;
    MelBank mel_;
    std::vector<float> win_;
    std::vector<std::complex<float>> spec_;
    std::vector<float> psd_;
    int usLo_, usHi_, spLo_, spHi_;

    bool calibrated_ = false;
    std::vector<std::vector<float>> calPsd_;
    std::vector<double> floorMedian_, floorMad_;

    std::deque<bool> aVotes_, bVotes_;
    std::vector<float> usEnergy_, livenessEnergy_, hfEnergy_;
    std::vector<float> liveWin_, hfWin_;
    std::vector<std::deque<float>> bandHistory_;
    double liveMedian_ = 0.0, hfMedian_ = 0.0;
    int bFrameCount_ = 0;
    int aHold_ = 0, bHold_ = 0, cooldown_ = 0;
    bool blockActive_ = false;
    std::vector<BlockEvent> events_;
    std::vector<FrameScore> scores_;
};

} // namespace hvc
