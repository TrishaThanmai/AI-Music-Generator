import os
from uuid import uuid4
import requests
import streamlit as st

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.models_labs import FileType, ModelsLabTools
from agno.utils.log import logger

# -------------------------
# Logging
# -------------------------
logger.setLevel("INFO")

# -------------------------
# Sidebar: API Keys
# -------------------------
st.sidebar.title("🔑 API Key Configuration")

openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password"
)

models_lab_api_key = st.sidebar.text_input(
    "Enter your ModelsLab API Key",
    type="password"
)

# -------------------------
# Main UI
# -------------------------
st.title("🎶 ModelsLab AI Music Generator")

prompt = st.text_area(
    "Enter your music generation prompt:",
    value="Generate a 30 second classical instrumental music piece",
    height=100,
)

# -------------------------
# Agent Initialization
# -------------------------
if openai_api_key and models_lab_api_key:

    agent = Agent(
        name="ModelsLab Music Agent",
        model=OpenAIChat(
            id="gpt-4o",
            api_key=openai_api_key
        ),
        show_tool_calls=True,
        tools=[
            ModelsLabTools(
                api_key=models_lab_api_key,
                wait_for_completion=True,
                file_type=FileType.MP3
            )
        ],
        markdown=True,
        description="You are an AI agent that generates high-quality instrumental music using the ModelsLab API.",
        instructions=[
            "When generating music, always use the `generate_media` tool with ultra-detailed prompts.",
            "Include these details:",
            "- Genre & style",
            "- Instruments used",
            "- Tempo & rhythm",
            "- Mood & emotional tone",
            "- Song structure (intro / verse / chorus / bridge / outro)",
            "Create clear, rich and structured prompts to guide the generator.",
            "Focus on producing complete instrumental music pieces suitable for listening."
        ],
    )

    # -------------------------
    # Generate Button
    # -------------------------
    if st.button("🎵 Generate Music"):

        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt first.")
            st.stop()

        with st.spinner("Generating music... Please wait 🎼"):

            try:
                music: RunOutput = agent.run(prompt)

                if not music.audio:
                    st.error("❌ No audio was returned from ModelsLab.")
                    st.stop()

                # -------------------------
                # Download Audio File
                # -------------------------
                url = music.audio[0].url

                response = requests.get(url)

                if not response.ok:
                    st.error(f"❌ Download failed: HTTP {response.status_code}")
                    st.stop()

                # Validate content type
                content_type = response.headers.get("Content-Type", "")
                if "audio" not in content_type:
                    st.error("❌ Invalid file type returned.")
                    st.write("Returned content-type:", content_type)
                    st.write("Audio URL:", url)
                    st.stop()

                # -------------------------
                # Save Audio
                # -------------------------
                save_dir = "audio_generations"
                os.makedirs(save_dir, exist_ok=True)

                filename = f"{save_dir}/music_{uuid4().hex}.mp3"

                with open(filename, "wb") as f:
                    f.write(response.content)

                # -------------------------
                # Play + Download
                # -------------------------
                st.success("✅ Music generated successfully!")

                audio_bytes = open(filename, "rb").read()
                st.audio(
                    audio_bytes,
                    format="audio/mp3"
                )

                st.download_button(
                    label="⬇️ Download MP3",
                    data=audio_bytes,
                    file_name="generated_music.mp3",
                    mime="audio/mp3"
                )

            except Exception as e:
                st.error("❌ An unexpected error occurred.")
                st.code(str(e))
                logger.error(f"Music generator failure: {e}")

else:
    st.sidebar.warning("⚠️ Please enter BOTH OpenAI and ModelsLab API keys to use the app.")
