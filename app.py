import streamlit as st
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

import tempfile
import os
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Multilingual Voice Translator",
    page_icon="🌍",
    layout="centered"
)


# ============================================================
# LANGUAGE LIST
# ============================================================

LANGUAGES = {
    "English": {
        "code": "en",
        "whisper": "en",
        "tts": "en"
    },

    "Telugu": {
        "code": "te",
        "whisper": "te",
        "tts": "te"
    },

    "Hindi": {
        "code": "hi",
        "whisper": "hi",
        "tts": "hi"
    },

    "Marathi": {
        "code": "mr",
        "whisper": "mr",
        "tts": "mr"
    },

    "Tamil": {
        "code": "ta",
        "whisper": "ta",
        "tts": "ta"
    },

    "Kannada": {
        "code": "kn",
        "whisper": "kn",
        "tts": "kn"
    },

    "Malayalam": {
        "code": "ml",
        "whisper": "ml",
        "tts": "ml"
    },

    "Bengali": {
        "code": "bn",
        "whisper": "bn",
        "tts": "bn"
    },

    "French": {
        "code": "fr",
        "whisper": "fr",
        "tts": "fr"
    },

    "German": {
        "code": "de",
        "whisper": "de",
        "tts": "de"
    },

    "Spanish": {
        "code": "es",
        "whisper": "es",
        "tts": "es"
    },

    "Japanese": {
        "code": "ja",
        "whisper": "ja",
        "tts": "ja"
    }
}


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_whisper_model():

    with st.spinner("🧠 Loading AI speech model..."):

        model = whisper.load_model("base")

    return model


# ============================================================
# HEADER
# ============================================================

st.title("🌍 AI Multilingual Voice Translator")

st.write(
    "🎤 Speak in one language → 🧠 AI understands → "
    "🌍 translates → 🔊 speaks the translation"
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_whisper_model()

except Exception as e:

    st.error("❌ Could not load the Whisper model.")

    st.code(str(e))

    st.info(
        "Make sure Whisper, PyTorch and FFmpeg are installed correctly."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

target_language = st.sidebar.selectbox(
    "🌍 Translate Into",
    list(LANGUAGES.keys())
)

st.sidebar.info(
    "The first time you run this application, "
    "Whisper may take some time to load."
)


# ============================================================
# TABS
# ============================================================

voice_tab, text_tab, history_tab = st.tabs(
    [
        "🎤 Voice Translator",
        "⌨️ Text Translator",
        "📜 History"
    ]
)


# ============================================================
# SESSION HISTORY
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# VOICE TRANSLATOR
# ============================================================

with voice_tab:

    st.header("🎤 Voice Translator")

    st.write(
        "Click the microphone button, speak, and then translate."
    )

    audio = st.audio_input(
        "🎤 Record your voice"
    )

    if audio is not None:

        st.audio(audio)

        st.success("✅ Audio recorded successfully!")

        translate_voice = st.button(
            "🌍 Translate Voice",
            type="primary",
            use_container_width=True
        )

        if translate_voice:

            # ------------------------------------------------
            # SAVE AUDIO TEMPORARILY
            # ------------------------------------------------

            temp_audio = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )

            temp_audio.write(audio.getvalue())

            temp_audio.close()

            try:

                # ------------------------------------------------
                # SPEECH TO TEXT
                # ------------------------------------------------

                with st.spinner(
                    "🎤 Converting speech into text..."
                ):

                    result = model.transcribe(
                        temp_audio.name
                    )

                original_text = result["text"].strip()

                detected_code = result.get(
                    "language",
                    "unknown"
                )

                # ------------------------------------------------
                # DISPLAY ORIGINAL TEXT
                # ------------------------------------------------

                st.subheader("📝 Recognized Speech")

                if original_text:

                    st.write(
                        f"**{original_text}**"
                    )

                else:

                    st.warning(
                        "⚠️ I could not understand the speech."
                    )

                    os.remove(temp_audio.name)

                    st.stop()

                # ------------------------------------------------
                # DETECTED LANGUAGE
                # ------------------------------------------------

                language_name = detected_code

                for name, information in LANGUAGES.items():

                    if information["whisper"] == detected_code:

                        language_name = name
                        break

                st.info(
                    f"🔎 Detected language: **{language_name}**"
                )

                # ------------------------------------------------
                # TRANSLATION
                # ------------------------------------------------

                target_code = LANGUAGES[
                    target_language
                ]["code"]

                with st.spinner(
                    "🌍 Translating..."
                ):

                    translated_text = GoogleTranslator(
                        source="auto",
                        target=target_code
                    ).translate(
                        original_text
                    )

                # ------------------------------------------------
                # DISPLAY TRANSLATION
                # ------------------------------------------------

                st.subheader(
                    f"🌍 {target_language} Translation"
                )

                st.success(
                    translated_text
                )

                # ------------------------------------------------
                # TEXT TO SPEECH
                # ------------------------------------------------

                with st.spinner(
                    "🔊 Creating spoken translation..."
                ):

                    tts = gTTS(
                        text=translated_text,
                        lang=LANGUAGES[
                            target_language
                        ]["tts"],
                        slow=False
                    )

                    audio_buffer = io.BytesIO()

                    tts.write_to_fp(
                        audio_buffer
                    )

                    audio_buffer.seek(0)

                # ------------------------------------------------
                # PLAY AUDIO
                # ------------------------------------------------

                st.subheader(
                    "🔊 Spoken Translation"
                )

                st.audio(
                    audio_buffer,
                    format="audio/mp3"
                )

                # ------------------------------------------------
                # DOWNLOAD AUDIO
                # ------------------------------------------------

                st.download_button(
                    label="⬇️ Download Translation Audio",
                    data=audio_buffer.getvalue(),
                    file_name="translation.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )

                # ------------------------------------------------
                # SAVE HISTORY
                # ------------------------------------------------

                st.session_state.history.append(
                    {
                        "Source Language": language_name,
                        "Target Language": target_language,
                        "Original": original_text,
                        "Translation": translated_text
                    }
                )

            except Exception as e:

                st.error(
                    "❌ Something went wrong."
                )

                st.code(str(e))

            finally:

                if os.path.exists(
                    temp_audio.name
                ):

                    os.remove(
                        temp_audio.name
                    )


# ============================================================
# TEXT TRANSLATOR
# ============================================================

with text_tab:

    st.header("⌨️ Text Translator")

    source_language = st.selectbox(
        "Source Language",
        list(LANGUAGES.keys()),
        key="text_source"
    )

    text_input = st.text_area(
        "Enter text to translate:",
        height=150,
        placeholder="Type something here..."
    )

    if st.button(
        "🌍 Translate Text",
        type="primary",
        use_container_width=True
    ):

        if not text_input.strip():

            st.warning(
                "⚠️ Please enter some text."
            )

        else:

            source_code = LANGUAGES[
                source_language
            ]["code"]

            target_code = LANGUAGES[
                target_language
            ]["code"]

            try:

                # ------------------------------------------------
                # TRANSLATE
                # ------------------------------------------------

                with st.spinner(
                    "🌍 Translating..."
                ):

                    translated_text = GoogleTranslator(
                        source=source_code,
                        target=target_code
                    ).translate(
                        text_input
                    )

                st.subheader(
                    f"🌍 {target_language}"
                )

                st.success(
                    translated_text
                )

                # ------------------------------------------------
                # TEXT TO SPEECH
                # ------------------------------------------------

                with st.spinner(
                    "🔊 Creating speech..."
                ):

                    tts = gTTS(
                        text=translated_text,
                        lang=LANGUAGES[
                            target_language
                        ]["tts"],
                        slow=False
                    )

                    audio_buffer = io.BytesIO()

                    tts.write_to_fp(
                        audio_buffer
                    )

                    audio_buffer.seek(0)

                st.subheader(
                    "🔊 Listen"
                )

                st.audio(
                    audio_buffer,
                    format="audio/mp3"
                )

                st.download_button(
                    label="⬇️ Download Audio",
                    data=audio_buffer.getvalue(),
                    file_name="translated_audio.mp3",
                    mime="audio/mpeg",
                    use_container_width=True
                )

                # ------------------------------------------------
                # SAVE HISTORY
                # ------------------------------------------------

                st.session_state.history.append(
                    {
                        "Source Language": source_language,
                        "Target Language": target_language,
                        "Original": text_input,
                        "Translation": translated_text
                    }
                )

            except Exception as e:

                st.error(
                    "❌ Translation failed."
                )

                st.code(str(e))


# ============================================================
# HISTORY
# ============================================================

with history_tab:

    st.header("📜 Translation History")

    if not st.session_state.history:

        st.info(
            "No translations yet."
        )

    else:

        for i, item in enumerate(
            reversed(
                st.session_state.history
            ),
            start=1
        ):

            with st.expander(
                f"Translation {i}"
            ):

                st.write(
                    f"**Source:** {item['Source Language']}"
                )

                st.write(
                    f"**Target:** {item['Target Language']}"
                )

                st.write(
                    f"**Original:** {item['Original']}"
                )

                st.write(
                    f"**Translation:** {item['Translation']}"
                )

        if st.button(
            "🗑️ Clear History"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Multilingual Voice Translator | "
    "Built with Python, Whisper, NLP and Streamlit"
)