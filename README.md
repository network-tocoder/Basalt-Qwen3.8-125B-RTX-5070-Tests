# Frontier-Class AI on an RTX 5070 (12 GB): Qwen3.8 Flash-Next 125B with Basalt

A 125B model building full apps on a mainstream 12 GB card. This repo has everything from the video: the exact model file, every command and setting, the benchmark, all 3 prompts (word for word), all 3 builds, and the full server logs.

![GPU](https://img.shields.io/badge/Tested%20on-RTX%205070%2012GB-76b900?style=for-the-badge&logo=nvidia&logoColor=white)
![Model](https://img.shields.io/badge/Model-Qwen-3.8.125--Flash%20321B-7c5cff?style=for-the-badge)
![Build](https://img.shields.io/badge/Build-Maya--S24%20(94.7%20GB)-0ea5e9?style=for-the-badge)
![RAM](https://img.shields.io/badge/RAM-60%20GB-f59e0b?style=for-the-badge)

[![Video](https://img.shields.io/badge/Video-NetworkCoder-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=PPE3OHQiL5M)


📺 **Video:** [Qwen 35B Is Now Insanely Fast on a 6GB GPU! (New llama.cpp)](https://www.youtube.com/watch?v=JECebDlkYmI) *(add the video link)*

---

## TL;DR

| What | Result |
|---|---|
| Reading prompts (16K prefill) | **2,823 tok/s** |
| Writing speed (board-style headline) | **~94 tok/s** |
| Writing normal text / code (real sessions) | **low 70s on short replies, ~62 tok/s sustained in long agent builds** |
| Context used | **128K configured; builds ran past 100K tokens with no slowdown cliff** |
| Builds | **3 complete single-file apps: 1,694 / 1,947 / 2,794 lines** |
| GPU memory | 12 GB card, with the experts in system RAM (`pcie = 0.0%`) |
| System RAM actually used | **~55 GB** (51.7 GiB loaded) |

---

## 1. Hardware

| Part | Spec |
|---|---|
| GPU | NVIDIA **RTX 5070, 12 GB** (Blackwell, sm_120) · driver 615.71.09 · CUDA 13.4 · 250 W power limit |
| CPU | AMD **Ryzen 9 5900X**, 12 cores (no AVX-512, so expert kernels use the AVX-2 path) |
| RAM | ~85 GiB available (only ~55 GB was used) |
| Storage | NVMe SSD |
| OS | Linux |

**Basalt needs:** an RTX 50-series / Blackwell GPU, CUDA 13, and an R580+ driver.

### How much RAM do you need?

| RAM | Verdict |
|---|---|
| **64 GB** | Realistic minimum for the 3-bit file |
| **96 GB** | Comfortable headroom |
| **~128 GB** | For the bigger 4-bit files |

If the experts don't fit in RAM, Basalt reads them from the SSD and speed drops sharply.

---

## 2. Engine and model

- **Engine:** [Basalt](https://github.com/jesdga95/basalt), built from source at commit `899e820a`. Basalt is built for Qwen3.8 Flash-Next on Blackwell.
- **Model file:** `Qwen3.8-FN-UD-IQ3_XXS.basalt` from Hugging Face `jesdga/Qwen3.8-Flash-Next-Basalt`.
  - **UD** means Unsloth Dynamic quantization; **IQ3_XXS** is 3-bit.
  - 81.6 GiB on disk, with 46.8 GiB of experts.
  - Only ~51.7 GiB is loaded. The image model and lookup tables stay on the SSD and are read only when needed.

### Available packs (size on disk)

| Pack | GiB | Notes |
|---|---:|---|
| 1-bit | 65.6 | |
| IQ2_XS | 74.4 | |
| **UD-IQ3_XXS** | **81.6** | **used here: the sweet spot for 12–16 GB cards** |
| IQ4_XS | 90.9 | |
| 4-bit Q4 | 107.4 | |
| NVFP4 (default) | 122.6 | aimed at 32 GB cards |
| Q6 | 161.2 | |
| 8-bit | 179.0 | |

### Speed features Basalt uses
- **Experts in system RAM, run on the CPU.** Expert misses are computed directly on the CPU instead of being copied over PCIe; the logs show `pcie=0.0%` for the whole session. This is why CPU speed matters.
- **MTP drafting** (speculative decoding, spec=8).
- **NVFP4** paths and **adaptive hot-expert swapping**.
- **`route-tail-skip` is OFF by default**, and I left it off, so the full model ran with nothing skipped.

---

## 3. Build Basalt

```bash
git clone https://github.com/jesdga95/basalt && cd basalt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j12
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # needed for the chat template (jinja2)
```

## 4. Download the model

```bash
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-IQ3_XXS.basalt --local-dir ~/models
```

Download the exact file name. A wildcard like `*IQ3_S*` matches nothing in this repo.

## 5. Start the server (OpenAI-compatible)

```bash
./bin/basalt serve ~/models/Qwen3.8-FN-UD-IQ3_XXS.basalt \
  --max-context 131072 --kv int8 \
  --alias qwen38-fn --host 127.0.0.1 --port 8080
```

On startup, Basalt:
- sets up a **128K context** with an **8-bit KV cache**;
- loads **~52 GB in about 9 seconds** (~6.5 GB/s);
- keeps **~49 GB of experts in system RAM**;
- keeps the conversation memory on the GPU, growing it as needed.

Quick check:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-fn","messages":[{"role":"user","content":"Say hi in 5 words"}]}'
```

Server defaults seen in the log: thinking budget 24,000 tokens, output cap 32,000.

---

## 6. Benchmark

### Community mode on 12 GB

```bash
basalt-bench --community --model Qwen3.8-FN-UD-IQ3_XXS.basalt
# basalt hc8: error: no VRAM for an 8-bit copy (3604480 bytes)
```

Community mode doesn't allow any flags, so that every row on the public board is comparable. On 12 GB it runs out of V RAM, which is why there are **no 12 GB cards on the board**.

### Same tests in normal mode with one change: a smaller memory reserve

```bash
CUDA_VISIBLE_DEVICES=0 ./bin/basalt-bench --model ~/models/Qwen3.8-FN-UD-IQ3_XXS.basalt --logs ~/bench_logs \
  -- --max-context 66661 --kv int8 --vram-reserve-mib 256
```

It uses the same prompts, the same context and the same 8-bit cache as community mode.

| Test | tok/s | Drafts accepted | Avg draft length |
|---|---:|---:|---:|
| pp16384 (reading a 16K prompt) | **2,823** | – | – |
| tg conversation | 72 | 69.2 % | 2.09 |
| tg structured (JSON) | 103 | 95.4 % | 6.66 |
| tg prose | 74 | 64.8 % | 1.70 |
| **Board-style headline (token-weighted)** | **~94** | | |

**Why the headline is higher than everyday speed:**
- About 80% of the test's tokens are JSON, where the drafts are accepted 95% of the time.
- Plain text and code land in the low 70s.
- This is true for every card on the board.

The engine was ready in 19 s, with 51.7 GiB loaded in 9 s. The raw output is in [`logs/bench_5070.log`](logs/bench_5070.log).

### Compared with the community board (decode tok/s, Linux unless noted)

These are single runs by different people with different files, so treat them as a rough guide.

| Card | V RAM | Decode tok/s |
|---|---|---|
| **RTX 5070 (this test)** | **12 GB** | **~94 (IQ3_XXS)** |
| RTX 5060 Ti | 16 GB | 102.8 (IQ3_XXS) · 137–159 (IQ3_S) · ~196 with extra cards |
| RTX 5070 Ti | 16 GB | ~152 Linux · 100–117 Windows |
| RTX 5080 | 16 GB | up to ~219 Linux · ~138 Windows |
| RTX 5090 | 32 GB | up to ~490 (IQ3_S) · 656 with two 5090s |
| RTX PRO 6000 | 96 GB | ~500–540 (whole model on the card) |

Prefill on 16K tokens: **5070 = 2,823 tok/s vs 5060 Ti = 1,796 tok/s (+57%)**. The 5070 has the stronger chip, while the 5060 Ti has 4 GB more V RAM.

**Patterns:**
1. V RAM matters most.
2. Linux beats Windows.
3. A second card helps.

---

## 7. Coding agent setup (Pi)

The builds used the **Pi** coding agent pointed at the local Basalt server. Save this as `~/.pi/agent/models.json` (also in [`config/models.json`](config/models.json)):

```json
{
  "providers": {
    "basalt": {
      "baseUrl": "http://localhost:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        { "id": "qwen38-fn", "name": "Qwen3.8 Flash Next (Basalt)", "contextWindow": 131072, "maxTokens": 16384 }
      ]
    }
  }
}
```

`maxTokens` was 32768 for builds 1 and 2. I lowered it to **16384** before build 3 so long sessions leave more room before compaction.

---

## 8. The three builds

Each build is a single prompt, with no hand-written code. The outputs are in [`projects/`](projects/).

| # | Build | Lines | Time | Tokens written | Tool calls | Writing speed | Notes |
|---|---|---:|---|---:|---:|---|---|
| 1 | 🎹 Beat Lab (web app) | **1,694** | 16 min | 64K | 37 | ~69 tok/s | 79 self-written test checks passed |
| 2 | 🗼 Neon Tower Defense (Three.js game) | **1,947** | ~60 min* | 165K | 121 | ~62.5 tok/s | A bot played all 10 waves; 5 bugs found and fixed; 2 auto-compactions |
| 3 | 🏯 Sky Market (3D scene) | **2,794** | 77 min | 194K | 161 | ~62.5 tok/s | 9 headless screenshots taken to check and fix the visuals |

\* Includes ~13 minutes idle after the model finished planning and stopped. One nudge got it writing again (see below).

The expert cache hit rate stayed between ~51% and ~57% across the builds.

### Writing speed vs context depth (across all builds)

| Context | tok/s |
|---|---:|
| 0–20K | 62.8 |
| 20–40K | 69.3 |
| 40–60K | 63.1 |
| 60–80K | 61.9 |
| 80–100K | 62.2 |
| 100–120K | 61.0 |

Speed is basically flat out to 120K.

---

## 9. Prompts (exact, word for word)

### Build 1: Beat Lab

```text
Build a complete music beat-maker web app in a single file called index.html, with no external assets or libraries.

- A 16-step drum sequencer with 8 tracks: kick, snare, clap, closed hat, open hat, tom, rim and cowbell. Synthesize every sound with the Web Audio API.
- Click cells to toggle steps, with a glowing playhead moving across the grid.
- Tempo slider (60–180 BPM), a swing control, and per-track volume, mute and solo.
- A 2-octave synth keyboard playable with the mouse and computer keys, with waveform select (sine, square, saw, triangle), filter cutoff and resonance knobs, and attack and release.
- A live audio visualizer (frequency bars plus waveform) that reacts to the music.
- Four pattern slots (A–D) you can switch between while playing, plus randomize and clear buttons.
- Save and load patterns to localStorage, and export the current pattern as JSON.
- 3 preset patterns (house, hip-hop, techno) that load on click.
- Dark studio look with neon accents, smooth knob dragging, responsive layout.

When finished, check the file is complete and valid, then give a short summary of the features and the number of lines.
```

### Build 2: Neon Tower Defense

```text
Build a complete tower defense game in a single HTML file using Three.js. A glowing neon grid map with a winding enemy path. Three tower types (laser, cannon, slow-field) with distinct looks, placement by clicking empty tiles, and two upgrade levels each. Ten waves of enemies with increasing speed and health, including a boss on wave 10. Gold earned per kill, lives lost when enemies reach the end. Particle explosions, tower range rings on hover, projectile trails. Sound effects with Web Audio API (no files). HUD with gold, lives, wave counter, start-wave and pause buttons, speed 1x/2x. Win and game-over screens with restart. Smooth 60fps, camera tilt with mouse drag. No external assets.
```

Follow-up nudge, sent once after it stopped at the plan:

```text
Go ahead and write the complete index.html now using the write tool, then check it and give the summary.
```

### Build 3: Sky Market

```text
Build a complete 3D scene in a single file called index.html using Three.js, with no other external assets.

The scene: a floating mountain marketplace in the sky, in a soft Chinese ink-painting style.
- A rocky island with a stone plaza and five small shops with dark tiled curved roofs and white walls.
- Wooden bridges and stairs leading to a smaller side island.
- Twisted pine trees on the cliffs, and layered misty mountain ranges in the background.

Day-to-night cycle:
- The sun moves across the sky and the fog colour shifts.
- At dusk, paper lanterns along the roofs and paths light up with a warm glow and soft point lights. Windows glow and fireflies appear.
- Birds glide around the island.

Interface:
- A side panel lists the five shops. Clicking one flies the camera to it and opens a small info card.
- Controls: orbit, zoom, reset view, fullscreen, a time-of-day slider and a mist toggle.

Quality: smooth 60fps, soft shadows, and gentle ambient sound via the Web Audio API (wind, plus wind chimes at night).

When finished, check the file is complete and valid, then give a short summary of the features and the number of lines.
```

### Run them yourself
Open any `projects/*/index.html` in a browser. The Three.js builds load Three.js from a CDN, so you need to be online. Click once to enable sound.

---

## 10. Tips and notes

- **12 GB works.** You'll need normal-mode bench flags (`--vram-reserve-mib 256`), since community mode has no room for its extra copies on 12 GB.
- **No AVX-512?** That's fine; Basalt falls back to AVX-2. A newer CPU with AVX-512 should be a little faster on the expert side.
- **Expect ~60–70 tok/s for real coding**, with ~94 as the board-style number. Long agent sessions held ~62 tok/s past 100K context.
- **Windows works**, but Linux was faster on every card compared.
- **More V RAM = more speed.** If you're buying a card for this, V RAM is where the money should go. A 5060 Ti 16 GB is the value pick.
- **Older RTX, AMD or Mac:** Basalt is Blackwell-only. On those, look at Strata (see my earlier 125B video).

---

## 11. Repo layout

```
README.md
config/models.json          Pi provider config (no keys)
logs/basalt_server.log      full server log for all three builds
logs/bench_5070.log         raw benchmark output
logs/nvidia-smi.txt         GPU / driver snapshot
projects/beat-lab/          index.html
projects/tower-defense/     index.html
projects/sky-market/        index.html + screenshots/ (the model's own headless checks)
```

---

## Credits

- **Basalt** by jesdga95: https://github.com/jesdga95/basalt
- **Model packs:** `jesdga/Qwen3.8-Flash-Next-Basalt`, built on Unsloth Dynamic quants of Qwen3.8 Flash-Next.
- **Pi** coding agent.

Drop your GPU, RAM and tok/s in the video comments, and tell me what I should build next.
