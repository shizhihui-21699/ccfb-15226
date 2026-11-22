// Evaluation harness: synthesizes the full test set, runs the streaming
// detector per microphone profile, and reports detection rate (attacks) and
// false-block rate (benign audio), plus a real-time cost benchmark.
#include "detector.hpp"
#include "synth.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <chrono>

using namespace hvc;

static constexpr double kFs = 48000.0;
static constexpr double kLead = 1.0;    // seconds of ambient lead-in (calib)
static constexpr double kBody = 2.5;    // seconds of event content
static constexpr int    kPerClass = 25; // clips per mic per class

struct ClassDef {
    const char* name;
    bool attack;
    std::function<std::vector<float>(Rng&)> gen;
};

static std::vector<float> genUAttack(Rng& r) {
    double fc = r.uniform(17500, 22500);
    float lvl = (float)r.uniform(0.05, 0.25);
    return ultrasonicAttack(r, kFs, kBody, lvl, fc);
}
static std::vector<float> genCommander(Rng& r) {
    // Realistic CommanderSong injection: command RMS a few dB below the
    // music bed (the published attacks that ASR still transcribes operate in
    // roughly -5..+2 dB; far below that the command is physically masked).
    double db = r.uniform(2, 7);
    // Two replay path profiles: telephone-grade and small-speaker grade.
    // Both are band-pass: telephone ~ [300,3400], small speaker ~ [180,5500].
    bool telephone = r.uniform() < 0.5;
    double low = telephone ? 300.0 : 180.0;
    double band = telephone ? 3400.0 : 5500.0;
    return commanderSongClip(r, kFs, kBody, 0.15f, db, low, band);
}
static std::vector<float> genUSWhine(Rng& r) {
    return ultrasonicWhine(r, kFs, kBody, (float)r.uniform(0.05, 0.2));
}
static std::vector<float> genFar(Rng& r) {
    return farFieldSpeech(r, kFs, kBody);
}
static std::vector<float> genFarNoisy(Rng& r) {
    return farFieldNoisy(r, kFs, kBody);
}
static std::vector<float> genSpMusic(Rng& r) {
    double db = r.uniform(3, 9);           // user voice above the music
    return speechWithMusic(r, kFs, kBody, db);
}
static std::vector<float> genMusic(Rng& r) {
    return musicOnly(r, kFs, kBody);
}
static std::vector<float> genNoise(Rng& r) {
    return noiseOnly(r, kFs, kBody);
}
static std::vector<float> genQuiet(Rng& r) {
    return whiteNoise(r, (size_t)(kBody * kFs), 0.004f);
}

static const ClassDef kClasses[] = {
    {"ultrasonic_attack", true,  genUAttack},
    {"commander_song",    true,  genCommander},
    {"ultrasonic_whine",  false, genUSWhine},
    {"farfield_speech",   false, genFar},
    {"farfield_noisy",    false, genFarNoisy},
    {"speech_over_music", false, genSpMusic},
    {"music_only",        false, genMusic},
    {"ambient_noise",     false, genNoise},
    {"near_silence",      false, genQuiet},
};
static constexpr int kNClasses = (int)(sizeof(kClasses) / sizeof(kClasses[0]));

struct ClassStat { int n = 0, blocked = 0; char channel[8] = ""; };

// Wilson score interval for a binomial proportion.
static void wilson(int x, int n, double& lo, double& hi) {
    double p = (double)x / n, z = 1.96;
    double d = 1 + z * z / n;
    double c = p + z * z / (2 * n);
    double e = z * std::sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    lo = (c - e) / d; hi = (c + e) / d;
}

int main(int argc, char** argv) {
    bool dumpCsv = argc > 1 && std::string(argv[1]) == "--csv";
    const char* csvPath = argc > 2 ? argv[2] : "scores.csv";
    FILE* csv = nullptr;
    if (dumpCsv) {
        csv = std::fopen(csvPath, "w");
        std::fprintf(csv, "clip,label,mic,frame,usRatio,usAbove,usAm,usPeak,"
                          "flatness,modSyll,speechRatio,liveness,hfCons,"
                          "liveMedian,hfMedian,aRaw,bRaw\n");
    }

    const int mics = 4;
    std::map<std::string, ClassStat> stats;
    double totalFrameUs = 0;
    long totalFrames = 0;
    long clipId = 0;

    for (int mic = 0; mic < mics; ++mic) {
        // One detector per mic: calibration on that mic's ambient floor.
        HiddenCommandDetector det;
        // Calibration signal: 1 s of that mic's self-noise / quiet room.
        Rng calRng(0xC0FFEEULL + mic * 7919);
        auto calib = whiteNoise(calRng, (size_t)(kLead * kFs), 0.004f);
        calib = applyMic(calib, kFs, mic, 0xBEEF00ULL + mic);
        // zero-pad one FFT window at the front
        std::vector<float> calStream(det.config().fftSize, 0.0f);
        calStream.insert(calStream.end(), calib.begin(), calib.end());
        for (size_t pos = 0; pos + det.config().hop <= calStream.size();
             pos += det.config().hop)
            det.calibrateFrame(&calStream[pos]);
        det.finishCalibration();

        for (int ci = 0; ci < kNClasses; ++ci) {
            const auto& cd = kClasses[ci];
            for (int rep = 0; rep < kPerClass; ++rep) {
                ++clipId;
                Rng r(0x1000 + mic * 100000 + ci * 1000 + rep * 37);
                auto body = cd.gen(r);
                auto throughMic = applyMic(body, kFs, mic,
                                            0x5151 + rep * 131 + mic * 17);
                std::vector<float> stream(det.config().fftSize, 0.0f);
                stream.insert(stream.end(), throughMic.begin(),
                              throughMic.end());
                det.resetRuntime();

                auto t0 = std::chrono::steady_clock::now();
                det.processStream(stream);
                auto t1 = std::chrono::steady_clock::now();
                int events = (int)det.events().size();
                totalFrameUs += std::chrono::duration<double,
                                     std::micro>(t1 - t0).count();
                totalFrames += (long)det.scores().size();

                auto& st = stats[cd.name];
                st.n++;
                if (events > 0) {
                    st.blocked++;
                    char ch = det.events().front().channel;
                    std::snprintf(st.channel, sizeof(st.channel), "%c", ch);
                }
                if (csv) {
                    const auto& sc = det.scores();
                    for (size_t f = 0; f < sc.size(); ++f) {
                        const auto& s = sc[f];
                        std::fprintf(csv,
                            "%ld,%s,%d,%zu,"
                            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                            "%.4f,%.4f,%d,%d\n",
                            clipId, cd.name, mic, f,
                            s.usRatio, s.usAboveFloor, s.usAmDepth, s.usPeakiness,
                            s.flatness, s.modSyll, s.speechRatio,
                            s.livenessMod, s.hfConsonant,
                            s.liveMedian, s.hfMedian,
                            (int)s.aRaw, (int)s.bRaw);
                    }
                }
            }
        }
    }
    if (csv) std::fclose(csv);

    // ---------------- report ----------------
    std::printf("================ Hidden-voice command detector evaluation ================\n");
    std::printf("fs=%.0f Hz  fft=%d  hop=%d (%.2f ms/frame)  clips/class=%d x %d mics\n\n",
                kFs, DetectorConfig().fftSize, DetectorConfig().hop,
                1000.0 * DetectorConfig().hop / kFs, kPerClass, mics);

    int attN = 0, attB = 0, benN = 0, benB = 0;
    std::printf("%-20s %8s %8s %10s %s\n", "class", "n", "blocked", "rate",
                "95%% Wilson CI");
    std::printf("------------------------------------------------------------------------\n");
    for (int ci = 0; ci < kNClasses; ++ci) {
        const auto& st = stats[kClasses[ci].name];
        double lo, hi; wilson(st.blocked, st.n, lo, hi);
        std::printf("%-20s %8d %8d %9.1f%%  [%.1f%%, %.1f%%]%s %s\n",
                    kClasses[ci].name, st.n, st.blocked,
                    100.0 * st.blocked / st.n, 100 * lo, 100 * hi,
                    st.blocked ? "ch=" : "    ",
                    st.blocked ? st.channel : "");
        if (kClasses[ci].attack) { attN += st.n; attB += st.blocked; }
        else                     { benN += st.n; benB += st.blocked; }
    }
    std::printf("------------------------------------------------------------------------\n");
    double lo, hi;
    wilson(attB, attN, lo, hi);
    std::printf("\nATTACK detection rate : %d/%d = %.2f%%  (95%% CI %.2f%% - %.2f%%)\n",
                attB, attN, 100.0 * attB / attN, 100 * lo, 100 * hi);
    wilson(benB, benN, lo, hi);
    std::printf("Benign false-block    : %d/%d = %.2f%%  (95%% CI %.2f%% - %.2f%%)\n",
                benB, benN, 100.0 * benB / benN, 100 * lo, 100 * hi);

    double perFrameUs = totalFrameUs / std::max(1L, totalFrames);
    double frameMs = DetectorConfig().hop / kFs * 1000.0;
    std::printf("\nReal-time cost: %.1f us/frame (frame budget %.1f ms) -> "
                "%.1f%% of one core, x%.0f real-time headroom\n",
                perFrameUs, frameMs, 100.0 * perFrameUs / (frameMs * 1000.0),
                frameMs * 1000.0 / perFrameUs);
    if (dumpCsv) std::printf("\nPer-frame scores written to %s\n", csvPath);
    return 0;
}
