import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "t5-small"

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
        inputs = tokenizer.encode(text, return_tensors='pt')
        outputs = model.generate(inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

def main():
    st.title('Language Translation App')

    # load the model and tokenizer
    model, tokenizer = load_model(MODEL_NAME)

    target_language = st.selectbox("Select target language", ["French", "Romanian", "German"])
    text = st.text_input("Enter text to translate")

    st.markdown('<style>body{background-color: white}</style>', unsafe_allow_html=True)
    st.markdown('<style>h1{color: #336699;}</style>', unsafe_allow_html=True)
    st.markdown('<style>label{color: #336699;}</style>', unsafe_allow_html=True)
    st.markdown('<style>button{background-color: #336699; color: white;}</style>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Submit'):
            translated_text = translate(text, model, tokenizer, target_language)
            st.text_area("Translated Text", translated_text, height=200)
    with col2:
        if st.button('Clear'):
            st.empty()

if __name__ == "__main__":
    main()