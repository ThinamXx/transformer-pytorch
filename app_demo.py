import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Thinam/lang-translate"


def load_model(model_name):
    """
    Loading the model and tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate(text, model, tokenizer, target_language):
    """
    Translate the input text to the target language.
    """
    try:
        text = f"{target_language} {text}"
        inputs = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""


def main():
    st.title("Language Translation App")

    model, tokenizer = load_model(MODEL_NAME)

    target_language = "German"

    input_text = st.text_area("Enter text to translate", height=200, key="input_text")
    output_text = st.empty()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Submit"):
            translated_text = translate(input_text, model, tokenizer, target_language)
            output_text.text_area(
                "Translated Text", translated_text, height=200, key="output_text"
            )

    with col2:
        if st.button("Clear"):
            input_text = ""
            output_text.text_area("Translated Text", value="")


if __name__ == "__main__":
    main()
