import os
from uuid import uuid4
import requests
import streamlit as st

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.models_labs import ModelsLabTools
from agno.utils.log import logger

logger.setLevel("INFO")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🔑 API Key Configuration")

openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
models_lab_api_key = st.sidebar.text_input("Enter ModelsLab API Key", type="password")

# -------------------------
# UI
# -------------------------
st.title("🎶 ModelsLab AI Music Generator")

prompt = st.text_area(
    "Enter your music prompt",
    "Generate a cinematic classical instrumental music piece",
    height=120,
)

# -------------------------
# AGENT INIT (MAX SAFE)
# -------------------------
if openai_api_key and models_lab_api_key:

    agent = Agent(
        name="ModelsLab Music Agent",
        model=OpenAIChat(
            id="gpt-4o",
            api_key=openai_api_key,
        ),
        tools=[
            ModelsLabTools(
                api_key=models_lab_api_key,
                wait_for_completion=True   # ✅ safe param
                # ❌ file_type removed to avoid crashes
            )
        ],
        system_prompt=(
            "You are a music generation assistant. "
            "Always produce detailed prompts containing "
            "genre, tone, instruments, tempo, and structure "
            "to generate professional instrumental music."
        )
    )

    # -------------------------
    # Generate Audio
    # -------------------------
    if st.button("🎵 Generate Music"):

        if not prompt.strip():
            st.warning("Please enter a prompt.")
            st.stop()

        with st.spinner("Generating music..."):

            try:
                result: RunOutput = agent.run(prompt)

                if not result.audio:
                    st.error("❌ No audio returned from ModelsLab")
                    st.stop()

                url = result.audio[0].url

                response = requests.get(url)

                if not response.ok:
                    st.error("Audio download failed")
                    st.stop()

                # -------------------------
                # Save audio
                # -------------------------
                os.makedirs("audio_generations", exist_ok=True)

                filename = f"audio_generations/music_{uuid4().hex}.mp3"
                with open(filename, "wb") as f:
                    f.write(response.content)

                audio_bytes = open(filename, "rb").read()

                # -------------------------
                # Playback
                # -------------------------
                st.success("✅ Music Generated Successfully!")
                st.audio(audio_bytes, format="audio/mp3")

                st.download_button(
                    "⬇️ Download MP3",
                    audio_bytes,
                    file_name="generated_music.mp3",
                    mime="audio/mp3",
                )

            except Exception as e:
                st.error("Unhandled generation error:")
                st.code(str(e))
                logger.error(e)

else:
    st.sidebar.warning("⚠️ Enter both API keys to begin.")
