// Generates one example WAV per class (through mic profile 1) so the
// synthesized signals can be inspected or played back.
#include "synth.hpp"
#include <cstdio>

using namespace hvc;

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : "data";
    constexpr double fs = 48000.0, sec = 2.5;
    struct Item { const char* name; std::vector<float> (*fn)(Rng&); };
    auto us = [](Rng& r) { return ultrasonicAttack(r, fs, sec, 0.18f, 20000); };
    auto cs = [](Rng& r) { return commanderSongClip(r, fs, sec, 0.15f, 8, 250.0, 4000.0); };
    auto wh = [](Rng& r) { return ultrasonicWhine(r, fs, sec, 0.15f); };
    auto ff = [](Rng& r) { return farFieldSpeech(r, fs, sec); };
    auto fn = [](Rng& r) { return farFieldNoisy(r, fs, sec); };
    auto sm = [](Rng& r) { return speechWithMusic(r, fs, sec, 6); };
    auto mu = [](Rng& r) { return musicOnly(r, fs, sec); };
    auto nz = [](Rng& r) { return noiseOnly(r, fs, sec); };
    Item items[] = {
        {"ultrasonic_attack", +us}, {"commander_song", +cs},
        {"ultrasonic_whine", +wh},  {"farfield_speech", +ff},
        {"farfield_noisy", +fn},    {"speech_over_music", +sm},
        {"music_only", +mu},        {"ambient_noise", +nz},
    };
    int mic = 1;
    for (auto& it : items) {
        Rng r(std::hash<std::string>{}(it.name) ^ 0xABCDEF);
        auto x = it.fn(r);
        x = applyMic(std::move(x), fs, mic, 0x777 + mic);
        char path[256];
        std::snprintf(path, sizeof(path), "%s/%s.wav", dir, it.name);
        if (!writeWav(path, x, (int)fs)) { std::perror(path); return 1; }
        std::printf("wrote %s (%.2f s, rms=%.4f)\n", path, sec, rmsOf(x));
    }
    return 0;
}
