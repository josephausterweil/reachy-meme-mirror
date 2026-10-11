---
title: Meme Mirror
emoji: 🪞
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: apache-2.0
short_description: Turn memes into Reachy Mini body language
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Meme Mirror

Turn memes into expressions for your Reachy Mini. Drop a meme into the app's page. An AI works out the feeling
behind it and proposes several ways Reachy could show it with its head, antennas and body. You compare them,
play the best ones on the robot, rate them, and ask for better ones. It learns your taste over time.

**See it in action:** <https://josephausterweil.github.io/reachy-meme-mirror/>
· **Code:** <https://github.com/josephausterweil/reachy-meme-mirror>

## After installing

1. Start **meme_mirror** from the dashboard and open its page, either with the app's settings button or at
   `http://reachy-mini.local:8043`.
2. Click **⚙ LLM** and choose an AI model. The app runs on the robot, so pick one it can reach:
   - **an API key:** Anthropic, OpenAI, Google Gemini or OpenRouter. Paste it in the panel;
   - **a model on a computer on your network:** Ollama, LM Studio, vLLM or similar. Choose
     "Other OpenAI-compatible server" and enter its address, e.g. `http://192.168.1.20:11434/v1`.
     Make sure that server accepts connections from your network;
   - **your Claude Code login on your computer:** run the bridge described in the GitHub README.

   Click **Test** to check it.
3. Drop in a meme and click **Propose expressions**.

**Optional:** learned "binh" motion needs PyTorch. Click **Install** next to *binh* on the page. It installs the
CPU build, about 0.9 GB, and a round takes about a minute on the robot. Without it, the binh slots are filled
by AI-written motion.

Previews on the robot are simple drawings plus a 3D viewer in your browser (drag to rotate). For simulator videos,
run the full version on a computer instead (see GitHub).

Your memes, notes and the prompts go to whichever AI provider you choose. With a model on your own network,
they stay on your network.

## Credits

Apache-2.0. Builds on Pollen Robotics' Reachy Mini SDK and emotions library, Pham Tuan Binh's
reachy-motion-generator and 3D viewer (with 8bitkick's 3D model), and community moves by Anne-Charlotte and
stellaaaa.
